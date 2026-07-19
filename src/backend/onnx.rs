//! Generic waveform-to-waveform ONNX backend using the pure-Rust tract runtime.
//!
//! The model must have exactly one `float32` input and one `float32` output.
//! Accepted layouts are `[batch, samples]` and `[batch, channels, samples]`;
//! denoize supplies a batch and channel size of one and processes file channels
//! independently. The output must contain at least as many samples as the model
//! input. Models operating on spectra or requiring an iterative sampler need a
//! dedicated adapter and are deliberately rejected by this backend.

use super::OnnxModelConfig;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;

pub fn process(
    channels: &[Vec<f64>],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<Vec<f64>>, String> {
    if config.sample_rate == 0 {
        return Err("ONNX model sample rate must be greater than zero".into());
    }
    if !config.path.is_file() {
        return Err(format!(
            "ONNX model does not exist or is not a file: {}",
            config.path.display()
        ));
    }

    channels
        .iter()
        .map(|channel| process_channel(channel, input_sample_rate, config))
        .collect()
}

fn process_channel(
    input: &[f64],
    input_sample_rate: u32,
    config: &OnnxModelConfig,
) -> Result<Vec<f64>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let model_input = resample_linear(input, input_sample_rate, config.sample_rate);
    let model_output = run_model(&model_input, config)?;
    let mut output = resample_linear(&model_output, config.sample_rate, input_sample_rate);
    output.truncate(input.len());
    output.resize(input.len(), 0.0);
    Ok(output)
}

fn run_model(input: &[f64], config: &OnnxModelConfig) -> Result<Vec<f64>, String> {
    let mut model = tract_onnx::onnx()
        .model_for_path(&config.path)
        .map_err(|e| format!("failed to load ONNX model {}: {e}", config.path.display()))?;

    if model.input_outlets().map_err(tract_error)?.len() != 1
        || model.output_outlets().map_err(tract_error)?.len() != 1
    {
        return Err("ONNX waveform model must have exactly one input and one output".into());
    }

    let rank = model
        .input_fact(0)
        .map_err(tract_error)?
        .shape
        .rank()
        .concretize()
        .ok_or_else(|| "ONNX model input rank must be known".to_string())?;
    let shape: TVec<usize> = match rank {
        2 => tvec!(1, input.len()),
        3 => tvec!(1, 1, input.len()),
        other => {
            return Err(format!(
                "unsupported ONNX input rank {other}; expected [batch, samples] or [batch, channels, samples]"
            ));
        }
    };
    model
        .set_input_fact(0, f32::fact(shape.clone()).into())
        .map_err(tract_error)?;
    model
        .set_output_fact(0, f32::fact(shape.clone()).into())
        .map_err(tract_error)?;

    let runnable = model
        .into_optimized()
        .and_then(|model| model.into_runnable())
        .map_err(tract_error)?;
    let samples: Vec<f32> = input.iter().map(|&sample| sample as f32).collect();
    let tensor = Tensor::from_shape(&shape, &samples).map_err(tract_error)?;
    let outputs = runnable
        .run(tvec!(tensor.into_tvalue()))
        .map_err(tract_error)?;
    let output = outputs[0].to_array_view::<f32>().map_err(tract_error)?;

    if output.len() < input.len() {
        return Err(format!(
            "ONNX model returned {} samples for an input of {}; output must not be shorter",
            output.len(),
            input.len()
        ));
    }
    let result: Vec<f64> = output
        .iter()
        .take(input.len())
        .map(|&sample| sample as f64)
        .collect();
    if result.iter().any(|sample| !sample.is_finite()) {
        return Err("ONNX model returned a non-finite audio sample".into());
    }
    Ok(result)
}

fn tract_error(error: impl std::fmt::Display) -> String {
    format!("ONNX inference failed: {error:#}")
}

pub(super) fn resample_linear(input: &[f64], from_rate: u32, to_rate: u32) -> Vec<f64> {
    if input.is_empty() || from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (input.len() as f64 * ratio).round() as usize;
    (0..output_len)
        .map(|index| {
            let source = index as f64 / ratio;
            let left = source.floor() as usize;
            let fraction = source - left as f64;
            let a = input.get(left).copied().unwrap_or(0.0);
            let b = input.get(left + 1).copied().unwrap_or(a);
            a + fraction * (b - a)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use tract_onnx::pb::{
        tensor_proto, tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto,
        OperatorSetIdProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };

    #[test]
    fn rejects_missing_model() {
        let config = OnnxModelConfig {
            path: std::path::PathBuf::from("definitely-missing-model.onnx"),
            sample_rate: 16_000,
        };
        let error = process(&[vec![0.0]], 16_000, &config).unwrap_err();
        assert!(error.contains("does not exist"));
    }

    #[test]
    fn round_trip_resampling_preserves_requested_length() {
        let input: Vec<f64> = (0..441).map(|index| index as f64 / 441.0).collect();
        let at_16k = resample_linear(&input, 44_100, 16_000);
        let restored = resample_linear(&at_16k, 16_000, 44_100);
        assert_eq!(at_16k.len(), 160);
        assert_eq!(restored.len(), input.len());
    }

    #[test]
    fn identity_waveform_model_runs_end_to_end() {
        let model = identity_model();
        let mut bytes = Vec::new();
        model.encode(&mut bytes).unwrap();
        let path = std::env::temp_dir().join(format!(
            "denoize-identity-{}-{}.onnx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();

        let config = OnnxModelConfig {
            path: path.clone(),
            sample_rate: 16_000,
        };
        let input = vec![vec![-0.5, 0.0, 0.25, 0.75]];
        let output = process(&input, 16_000, &config).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), input[0].len());
        for (actual, expected) in output[0].iter().zip(&input[0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    fn identity_model() -> ModelProto {
        let value_info = |name: &str| ValueInfoProto {
            name: name.into(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: tensor_proto::DataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: vec![dimension_value(1), dimension_parameter("samples")],
                    }),
                })),
            }),
            doc_string: String::new(),
        };
        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            producer_name: "denoize-test".into(),
            graph: Some(GraphProto {
                name: "identity-waveform".into(),
                node: vec![NodeProto {
                    input: vec!["input".into()],
                    output: vec!["output".into()],
                    name: "identity".into(),
                    op_type: "Identity".into(),
                    ..Default::default()
                }],
                input: vec![value_info("input")],
                output: vec![value_info("output")],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn dimension_value(value: i64) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
            denotation: String::new(),
        }
    }

    fn dimension_parameter(name: &str) -> tensor_shape_proto::Dimension {
        tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(name.into())),
            denotation: String::new(),
        }
    }
}

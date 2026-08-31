-- Exercise the same REAPER FX parameter API used by OSARA's parameters dialog.
--
-- Run with REAPER 7.79 or later and set DENOIZE_REAPER_RESULT to a writable
-- result path. The script deliberately uses TrackFX_SetParam (native range),
-- rather than reaching into the CLAP ABI directly.

local result_path = os.getenv("DENOIZE_REAPER_RESULT")
if result_path == nil or result_path == "" then
  error("DENOIZE_REAPER_RESULT is required")
end

local result = assert(io.open(result_path, "w"))

local function write_line(...)
  local fields = { ... }
  for index, field in ipairs(fields) do
    fields[index] = tostring(field)
  end
  result:write(table.concat(fields, "\t"), "\n")
  result:flush()
end

local track = reaper.InsertTrackAtIndex(0, true)
if track == nil then
  track = reaper.GetTrack(0, 0)
end
if track == nil then
  write_line("error", "unable to create test track")
  result:close()
  reaper.Main_OnCommand(40004, 0)
  return
end

local audio_path = os.getenv("DENOIZE_REAPER_AUDIO")
if audio_path ~= nil and audio_path ~= "" then
  reaper.SetOnlyTrackSelected(track)
  reaper.SetEditCurPos(0, false, false)
  reaper.InsertMedia(audio_path, 0)
  write_line(
    "media",
    audio_path,
    reaper.CountMediaItems(0),
    reaper.CountTrackMediaItems(track)
  )
end

local plugin_name = os.getenv("DENOIZE_REAPER_PLUGIN") or "denoize Neural"
local fx = reaper.TrackFX_AddByName(track, "CLAP: " .. plugin_name, false, -1)
if fx < 0 then
  fx = reaper.TrackFX_AddByName(track, plugin_name, false, -1)
end
if fx < 0 then
  write_line("error", plugin_name .. " was not discovered")
  local enum_match = string.lower(
    os.getenv("DENOIZE_REAPER_ENUM_MATCH") or "denoize"
  )
  local installed = 0
  while true do
    local found, name, identifier = reaper.EnumInstalledFX(installed)
    if not found then
      break
    end
    if string.find(string.lower(name), enum_match, 1, true) then
      write_line("installed", installed, name, identifier)
    end
    installed = installed + 1
  end
  result:close()
  reaper.Main_OnCommand(40004, 0)
  return
end

local _, fx_name = reaper.TrackFX_GetFXName(track, fx)
local parameter_count = reaper.TrackFX_GetNumParams(track, fx)
write_line("plugin", fx, fx_name, parameter_count)

local osara_style = os.getenv("DENOIZE_REAPER_OSARA_STYLE") == "1"

local function osara_target(parameter, before, minimum, maximum)
  local has_steps, step, _, large_step, is_toggle =
    reaper.TrackFX_GetParameterStepSizes(track, fx, parameter)
  if not has_steps or step == nil or step == 0 then
    step = (maximum - minimum) / 1000
    large_step = step * 20
  elseif large_step == nil or large_step == 0 then
    large_step = (maximum - minimum) / 50
    large_step = step * math.floor(large_step / step)
    if large_step == 0 then
      large_step = step
    end
  end

  local direction = before + step <= maximum and 1 or -1
  local target = before + (step * direction)
  local _, before_text = reaper.TrackFX_FormatParamValue(
    track,
    fx,
    parameter,
    before
  )
  if before_text ~= nil and before_text ~= "" then
    for steps = 1, 10000 do
      local candidate = before + (step * direction * steps)
      if candidate < minimum or candidate > maximum then
        break
      end
      local _, candidate_text = reaper.TrackFX_FormatParamValue(
        track,
        fx,
        parameter,
        candidate
      )
      target = candidate
      if candidate_text ~= before_text then
        break
      end
    end
  end

  return target, has_steps, step, large_step, is_toggle, before_text
end

local probes = {}
local verify_started = nil
local function verify_after_host_flush()
  if reaper.time_precise() - verify_started < 0.75 then
    reaper.defer(verify_after_host_flush)
    return
  end

  local failures = 0
  for _, probe in ipairs(probes) do
    local after = reaper.TrackFX_GetParam(track, fx, probe.parameter)
    local changed = math.abs(after - probe.before) > 1e-9
    if not probe.accepted or not changed then
      failures = failures + 1
    end
    write_line(
      "parameter",
      probe.parameter,
      probe.name,
      string.format("%.17g", probe.minimum),
      string.format("%.17g", probe.maximum),
      string.format("%.17g", probe.before),
      string.format("%.17g", probe.target),
      tostring(probe.accepted),
      string.format("%.17g", after),
      tostring(changed)
    )
  end

  write_line("summary", failures)
  result:close()
end

local function set_parameters()
  write_line(
    "host",
    reaper.GetPlayState(),
    reaper.TrackFX_GetEnabled(track, fx),
    reaper.TrackFX_GetOffline(track, fx)
  )
  local normalized = os.getenv("DENOIZE_REAPER_NORMALIZED") == "1"
  local plugin_parameter_count = tonumber(
    os.getenv("DENOIZE_REAPER_PLUGIN_PARAMETER_COUNT") or ""
  ) or math.max(0, parameter_count - 3)
  for parameter = 0, plugin_parameter_count - 1 do
    local _, name = reaper.TrackFX_GetParamName(track, fx, parameter)
    local before, minimum, maximum = reaper.TrackFX_GetParam(
      track, fx, parameter
    )
    local target
    if osara_style then
      local last_touched = reaper.TrackFX_SetNamedConfigParm(
        track,
        fx,
        "last_touched",
        tostring(parameter)
      )
      local focused = reaper.TrackFX_SetNamedConfigParm(
        track,
        fx,
        "focused",
        "1"
      )
      local has_steps, step, large_step, is_toggle, before_text
      target, has_steps, step, large_step, is_toggle, before_text = osara_target(
        parameter,
        before,
        minimum,
        maximum
      )
      write_line(
        "osara",
        parameter,
        name,
        tostring(last_touched),
        tostring(focused),
        tostring(has_steps),
        string.format("%.17g", step),
        string.format("%.17g", large_step),
        tostring(is_toggle),
        before_text or ""
      )
    else
      target = minimum + ((maximum - minimum) * 0.37)
      if name == "Bypass" or name == "Delta" or name == "Overload Fallback" then
        target = before == minimum and maximum or minimum
      end
      if math.abs(target - before) < 1e-9 then
        target = minimum + ((maximum - minimum) * 0.73)
      end
    end
    local accepted
    if normalized then
      accepted = reaper.TrackFX_SetParamNormalized(
        track,
        fx,
        parameter,
        (target - minimum) / (maximum - minimum)
      )
    else
      accepted = reaper.TrackFX_SetParam(track, fx, parameter, target)
    end
    probes[#probes + 1] = {
      parameter = parameter,
      name = name,
      minimum = minimum,
      maximum = maximum,
      before = before,
      target = target,
      accepted = accepted,
    }
  end

  verify_started = reaper.time_precise()
  reaper.defer(verify_after_host_flush)
end

local set_delay = tonumber(os.getenv("DENOIZE_REAPER_SET_DELAY") or "0") or 0
if set_delay <= 0 then
  set_parameters()
end

local play_delay = tonumber(os.getenv("DENOIZE_REAPER_PLAY_DELAY") or "0") or 0
local function start_playback()
  reaper.Main_OnCommand(1007, 0)
  write_line("play", reaper.GetPlayState())
end

if os.getenv("DENOIZE_REAPER_PLAY") == "1" then
  if play_delay <= 0 then
    start_playback()
  else
    local script_started = reaper.time_precise()
    local function play_after_delay()
      if reaper.time_precise() - script_started < play_delay then
        reaper.defer(play_after_delay)
        return
      end
      start_playback()
    end
    reaper.defer(play_after_delay)
  end
end

if set_delay > 0 then
  local play_started = reaper.time_precise()
  local function set_after_delay()
    if reaper.time_precise() - play_started < set_delay then
      reaper.defer(set_after_delay)
      return
    end
    set_parameters()
  end
  reaper.defer(set_after_delay)
end

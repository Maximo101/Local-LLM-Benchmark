"""Built-in tool-calling eval (BFCL-style, "FC" mode): OpenAI `tools` definitions, expected calls.

Categories (the first four as in the Berkeley Function-Calling Leaderboard, the last two after its V3 multi-turn set):
  simple       one tool offered, call it with the right arguments
  multiple     several tools offered (some deliberately close: forecast vs current weather, exchange rate vs
               conversion, turn off vs turn on), pick the right one
  parallel     the request needs two or three calls in the same reply
  irrelevance  tools are offered but none fits: the model must answer WITHOUT calling a tool
  missing      (v2.3.6) a tool fits but the user left out a required value (no e-mail address, no room, no repo):
               the model must ask, not call with a made-up value
  chained      (v2.3.6) the conversation already holds the model's earlier call and the tool's result; the model must
               use that result for the next call (or answer without a call when the result settles it)

An expected argument is a list of acceptable values; None in that list means the argument may be left out. An
argument the user did not give and that is not in the expectation at all is not checked. Values are compared
case-insensitively, numbers numerically, lists as unordered sets. An argument that is not in the tool's schema fails
the case (a made-up parameter would break a real agent).

ORDER MATTERS (v2.3.6). The cases are asked as a prefix of CASES: never fewer than the first 32 (TOOL_MIN_CASES in
config.py), more when a profile or funnel stage asks more rows (moderate 43, standard 96 = every case). The first
32 are the v2.2-v2.3.5 set (multiple-10 replaced, see below), so MIN_TOOLCALL_PCT is decided on them at the first
funnel stage and cached answers keep matching. New cases are only ever APPENDED; changing a case's tools, prompt or
history changes its result-cache key and it is asked again.

multiple-10 was "How do you say 'thank you' in German?" with translate_text offered. About half the models in a
22-model run answered it directly, which is as reasonable as the irrelevance cases' direct answers, so it
measured a coin-flip of style rather than tool use. multiple-10b asks for the tool explicitly.

These are written for this tool (no download, no licence question) and kept unambiguous: the point is to catch
chat-template / --jinja / tool-parser problems, weak argument filling and poor judgement about when to call, not to
rank frontier models.
"""
from __future__ import annotations


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}

TOOLS = {
    "get_weather": _fn("get_weather", "Current weather for a city.",
                       {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    "convert_currency": _fn("convert_currency", "Convert an amount between currencies (ISO codes).",
                            {"amount": N, "from_currency": S, "to_currency": S}, ["amount", "from_currency", "to_currency"]),
    "set_thermostat": _fn("set_thermostat", "Set the target temperature (Celsius) of a room's thermostat.",
                          {"room": S, "temperature": N, "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}},
                          ["room", "temperature"]),
    "turn_on_light": _fn("turn_on_light", "Turn on a Home Assistant light entity, optionally at a brightness (0-100).",
                         {"entity_id": S, "brightness": I}, ["entity_id"]),
    "search_web": _fn("search_web", "Search the web.", {"query": S, "max_results": I}, ["query"]),
    "create_calendar_event": _fn("create_calendar_event", "Create a calendar event.",
                                 {"title": S, "date": {"type": "string", "description": "YYYY-MM-DD"},
                                  "time": {"type": "string", "description": "HH:MM, 24-hour"},
                                  "duration_minutes": I}, ["title", "date"]),
    "send_email": _fn("send_email", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    "calculate_mortgage": _fn("calculate_mortgage", "Monthly repayment of a loan.",
                              {"principal": N, "annual_rate_pct": N, "years": I}, ["principal", "annual_rate_pct", "years"]),
    "get_stock_price": _fn("get_stock_price", "Latest price of a stock by ticker symbol.", {"ticker": S}, ["ticker"]),
    "add_todo": _fn("add_todo", "Add an item to the to-do list.",
                    {"text": S, "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                     "tags": {"type": "array", "items": S}}, ["text"]),
    "read_file": _fn("read_file", "Read a text file.", {"path": S}, ["path"]),
    "list_directory": _fn("list_directory", "List the files in a directory.", {"path": S}, ["path"]),
    "play_music": _fn("play_music", "Play music on a speaker.", {"artist": S, "song": S, "room": S}),
    "get_distance": _fn("get_distance", "Travel distance and time between two places.",
                        {"origin": S, "destination": S, "mode": {"type": "string", "enum": ["driving", "walking", "transit"]}},
                        ["origin", "destination"]),
    "translate_text": _fn("translate_text", "Translate text into another language.",
                          {"text": S, "target_language": S}, ["text", "target_language"]),
    "set_timer": _fn("set_timer", "Start a countdown timer.", {"minutes": I, "label": S}, ["minutes"]),
    # ---- v2.3.6 (the definitions above are part of cached answers' keys: never edit them, only add) ----------------
    "get_forecast": _fn("get_forecast", "Weather forecast for the coming days for a city.",
                        {"city": S, "days": I, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                        ["city", "days"]),
    "turn_off_light": _fn("turn_off_light", "Turn off a Home Assistant light entity.", {"entity_id": S}, ["entity_id"]),
    "lock_door": _fn("lock_door", "Lock a Home Assistant lock entity.", {"entity_id": S}, ["entity_id"]),
    "get_exchange_rate": _fn("get_exchange_rate", "Current exchange rate between two currencies (ISO codes). Does not "
                             "convert an amount.", {"from_currency": S, "to_currency": S}, ["from_currency", "to_currency"]),
    "search_files": _fn("search_files", "Find files whose names match a glob pattern anywhere under a directory "
                        "(recursive).", {"directory": S, "pattern": S}, ["directory", "pattern"]),
    "write_file": _fn("write_file", "Write text to a file, replacing its contents.", {"path": S, "content": S},
                      ["path", "content"]),
    "run_shell_command": _fn("run_shell_command", "Run a shell command on the server and return its output.",
                             {"command": S, "timeout_s": I}, ["command"]),
    "restart_container": _fn("restart_container", "Restart a Docker container by name.", {"name": S}, ["name"]),
    "get_container_logs": _fn("get_container_logs", "The last lines of a Docker container's log.",
                              {"name": S, "lines": I}, ["name"]),
    "book_restaurant": _fn("book_restaurant", "Book a table at a restaurant.",
                           {"restaurant": S, "date": {"type": "string", "description": "YYYY-MM-DD"},
                            "time": {"type": "string", "description": "HH:MM, 24-hour"}, "party_size": I},
                           ["restaurant", "date", "time", "party_size"]),
    "create_issue": _fn("create_issue", "Open an issue in a GitHub repository.",
                        {"repo": {"type": "string", "description": "owner/name"}, "title": S,
                         "labels": {"type": "array", "items": S}}, ["repo", "title"]),
    "set_volume": _fn("set_volume", "Set a speaker's volume (0-100).", {"room": S, "level": I}, ["room", "level"]),
    "get_calendar_events": _fn("get_calendar_events", "List the calendar events on a date.",
                               {"date": {"type": "string", "description": "YYYY-MM-DD"}}, ["date"]),
    "get_order_status": _fn("get_order_status", "Delivery status of an online order.", {"order_id": S}, ["order_id"]),
    "schedule_meeting": _fn("schedule_meeting", "Invite people to a meeting.",
                            {"title": S, "date": {"type": "string", "description": "YYYY-MM-DD"},
                             "time": {"type": "string", "description": "HH:MM, 24-hour"},
                             "attendees": {"type": "array", "items": {"type": "string", "description": "e-mail address"}}},
                            ["title", "date", "time", "attendees"]),
}


def _c(cid, cat, tools, prompt, expect, history=None):
    c = {"id": cid, "category": cat, "tools": tools, "prompt": prompt, "expect": expect}
    if history:
        c["history"] = history
    return c


def did(*steps):
    """The turn(s) before the one graded (chained cases): `steps` are (name, args, result) of calls the model already
    made, all in one assistant message, each followed by its tool result. Ids are 9 alphanumerics (some templates,
    e.g. Mistral's, insist on that)."""
    import json
    calls, results = [], []
    for i, (name, args, result) in enumerate(steps):
        cid = f"call{i + 1:05d}"
        calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})
        results.append({"role": "tool", "tool_call_id": cid, "name": name, "content": json.dumps(result)})
    return [{"role": "assistant", "content": "", "tool_calls": calls}] + results


def call(fn, /, **args):
    """An expected call. `fn` is positional-only so a tool can have an argument called `name`."""
    return {"name": fn, "args": args}


MEL = ["Melbourne", "Melbourne, Australia", "Melbourne, VIC", "Melbourne, AU"]


CASES = [
    # ---- simple -------------------------------------------------------------------------------------------------
    _c("simple-01", "simple", ["get_weather"], "What's the weather in Melbourne right now? Use Celsius.",
       [call("get_weather", city=["Melbourne", "Melbourne, Australia", "Melbourne, VIC", "Melbourne, AU"],
             unit=["celsius"])]),
    _c("simple-02", "simple", ["convert_currency"], "Convert 250 US dollars to euros.",
       [call("convert_currency", amount=[250], from_currency=["USD"], to_currency=["EUR"])]),
    _c("simple-03", "simple", ["set_thermostat"], "Set the bedroom thermostat to 21 degrees.",
       [call("set_thermostat", room=["bedroom"], temperature=[21], mode=[None, "auto", "heat"])]),
    _c("simple-04", "simple", ["turn_on_light"], "Turn on light.kitchen_main at 40% brightness.",
       [call("turn_on_light", entity_id=["light.kitchen_main"], brightness=[40])]),
    _c("simple-05", "simple", ["create_calendar_event"],
       "Add 'Dentist' to my calendar on 2026-12-03 at 14:30 for 45 minutes.",
       [call("create_calendar_event", title=["Dentist"], date=["2026-12-03"], time=["14:30"], duration_minutes=[45])]),
    _c("simple-06", "simple", ["calculate_mortgage"],
       "What's the monthly repayment on a 600000 loan at 6.1% over 30 years?",
       [call("calculate_mortgage", principal=[600000], annual_rate_pct=[6.1], years=[30])]),
    _c("simple-07", "simple", ["add_todo"], "Add 'renew car registration' to my to-do list as high priority, tagged car.",
       [call("add_todo", text=["renew car registration", "Renew car registration"], priority=["high"],
             tags=[["car"], None])]),
    _c("simple-08", "simple", ["set_timer"], "Start a 12 minute timer called pasta.",
       [call("set_timer", minutes=[12], label=["pasta", "Pasta"])]),
    _c("simple-09", "simple", ["get_stock_price"], "What's NVIDIA trading at? The ticker is NVDA.",
       [call("get_stock_price", ticker=["NVDA"])]),
    _c("simple-10", "simple", ["translate_text"], "Translate 'where is the train station' into Japanese.",
       [call("translate_text", text=["where is the train station", "Where is the train station",
                                     "Where is the train station?", "where is the train station?"],
             target_language=["Japanese", "ja", "jp"])]),
    # ---- multiple -----------------------------------------------------------------------------------------------
    _c("multiple-01", "multiple", ["get_weather", "get_distance", "search_web"],
       "How far is it to drive from Geelong to Ballarat?",
       [call("get_distance", origin=["Geelong"], destination=["Ballarat"], mode=["driving", None])]),
    _c("multiple-02", "multiple", ["read_file", "list_directory", "send_email"], "Show me the contents of /etc/hostname.",
       [call("read_file", path=["/etc/hostname"])]),
    _c("multiple-03", "multiple", ["read_file", "list_directory", "send_email"], "What files are in /var/log?",
       [call("list_directory", path=["/var/log", "/var/log/"])]),
    _c("multiple-04", "multiple", ["play_music", "turn_on_light", "set_timer"], "Play Thunderstruck by AC/DC in the garage.",
       [call("play_music", artist=["AC/DC", "ACDC", "AC DC"], song=["Thunderstruck"], room=["garage"])]),
    _c("multiple-05", "multiple", ["get_stock_price", "convert_currency", "search_web"],
       "How many Japanese yen is 80 Australian dollars?",
       [call("convert_currency", amount=[80], from_currency=["AUD"], to_currency=["JPY"])]),
    _c("multiple-06", "multiple", ["send_email", "create_calendar_event", "add_todo"],
       "Email sam@example.com with the subject 'Lunch' and say 'Running 10 minutes late.'",
       [call("send_email", to=["sam@example.com"], subject=["Lunch"],
             body=["Running 10 minutes late.", "Running 10 minutes late", "running 10 minutes late."])]),
    _c("multiple-07", "multiple", ["set_thermostat", "turn_on_light", "get_weather"],
       "Cool the living room down to 23 degrees.",
       [call("set_thermostat", room=["living room", "living_room", "livingroom"], temperature=[23], mode=["cool", None])]),
    _c("multiple-08", "multiple", ["search_web", "read_file", "get_weather"],
       "Search the web for 'llama.cpp n-cpu-moe flag' and give me 5 results.",
       [call("search_web", query=["llama.cpp n-cpu-moe flag"], max_results=[5])]),
    _c("multiple-09", "multiple", ["set_timer", "create_calendar_event", "add_todo"],
       "Remind me in 25 minutes to take the washing out (just a timer is fine).",
       [call("set_timer", minutes=[25], label=[None, "take the washing out", "washing", "Take the washing out"])]),
    # v2.3.6: replaces multiple-10 ("How do you say 'thank you' in German?"), which a model could fairly answer itself
    _c("multiple-10b", "multiple", ["translate_text", "search_web", "send_email"],
       "Use the translation tool to translate 'The printer on level 3 is out of toner.' into German.",
       [call("translate_text", text=["The printer on level 3 is out of toner."], target_language=["German", "de"])]),
    # ---- parallel -----------------------------------------------------------------------------------------------
    _c("parallel-01", "parallel", ["get_weather"], "What's the weather in Sydney and in Perth? Celsius please.",
       [call("get_weather", city=["Sydney", "Sydney, Australia", "Sydney, NSW"], unit=["celsius"]),
        call("get_weather", city=["Perth", "Perth, Australia", "Perth, WA"], unit=["celsius"])]),
    _c("parallel-02", "parallel", ["turn_on_light", "play_music"],
       "Turn on light.lounge at full brightness and play some Coldplay in the lounge.",
       [call("turn_on_light", entity_id=["light.lounge"], brightness=[100, None]),
        call("play_music", artist=["Coldplay"], room=["lounge"], song=[None])]),
    _c("parallel-03", "parallel", ["get_stock_price"], "Get me the prices for AAPL and MSFT.",
       [call("get_stock_price", ticker=["AAPL"]), call("get_stock_price", ticker=["MSFT"])]),
    _c("parallel-04", "parallel", ["set_timer", "add_todo"],
       "Set a 5 minute timer for tea, and add 'buy milk' to my to-do list.",
       [call("set_timer", minutes=[5], label=["tea", "Tea"]),
        call("add_todo", text=["buy milk", "Buy milk"], priority=[None], tags=[None])]),
    _c("parallel-05", "parallel", ["convert_currency"], "Convert 100 EUR to USD and 100 EUR to GBP.",
       [call("convert_currency", amount=[100], from_currency=["EUR"], to_currency=["USD"]),
        call("convert_currency", amount=[100], from_currency=["EUR"], to_currency=["GBP"])]),
    _c("parallel-06", "parallel", ["read_file"], "Read both /etc/hosts and /etc/fstab.",
       [call("read_file", path=["/etc/hosts"]), call("read_file", path=["/etc/fstab"])]),
    # ---- irrelevance --------------------------------------------------------------------------------------------
    _c("irrelevance-01", "irrelevance", ["get_weather", "convert_currency"], "What is 17 multiplied by 23?", []),
    _c("irrelevance-02", "irrelevance", ["send_email", "read_file"], "Who wrote the novel 'Pride and Prejudice'?", []),
    _c("irrelevance-03", "irrelevance", ["set_thermostat", "turn_on_light"], "Tell me a short joke about computers.", []),
    _c("irrelevance-04", "irrelevance", ["get_stock_price"], "Explain in one sentence what a stock dividend is.", []),
    _c("irrelevance-05", "irrelevance", ["get_distance", "play_music"], "What is the capital city of Canada?", []),
    _c("irrelevance-06", "irrelevance", ["create_calendar_event", "set_timer"], "How many days are there in a leap year?", []),
    # =========================================================================================================== v2.3.6
    # Appended cases, interleaved so any prefix is a mix of categories. The first 11 are what a 43-row (moderate)
    # stage adds to the core 32; all 64 are asked from 96 rows (standard) up.
    # ---- 33-43 ----------------------------------------------------------------------------------------------------
    _c("simple-12", "simple", ["create_calendar_event"],
       "Today is Tuesday 2026-03-10. Put 'Car service' in my calendar for the day after tomorrow at 8am.",
       [call("create_calendar_event", title=["Car service"], date=["2026-03-12"], time=["08:00", "8:00"],
             duration_minutes=[None])]),
    _c("multiple-13", "multiple", ["convert_currency", "get_exchange_rate", "get_stock_price"],
       "What's the current US dollar to Japanese yen exchange rate?",
       [call("get_exchange_rate", from_currency=["USD"], to_currency=["JPY"])]),
    _c("parallel-07", "parallel", ["get_weather"], "What's the weather in Darwin, Cairns and Alice Springs? Celsius please.",
       [call("get_weather", city=["Darwin", "Darwin, Australia", "Darwin, NT"], unit=["celsius"]),
        call("get_weather", city=["Cairns", "Cairns, Australia", "Cairns, QLD"], unit=["celsius"]),
        call("get_weather", city=["Alice Springs", "Alice Springs, Australia", "Alice Springs, NT"], unit=["celsius"])]),
    _c("irrelevance-07", "irrelevance", ["set_thermostat", "get_weather"], "In general terms, what does a thermostat do?", []),
    _c("missing-01", "missing", ["send_email"], "Email my landlord that the hot water is broken.", []),
    _c("chained-01", "chained", ["get_stock_price", "convert_currency"],
       "What is one Apple share (AAPL) worth in Australian dollars?",
       [call("convert_currency", amount=[190.5], from_currency=["USD"], to_currency=["AUD"])],
       did(("get_stock_price", {"ticker": "AAPL"}, {"ticker": "AAPL", "price": 190.5, "currency": "USD"}))),
    _c("simple-16", "simple", ["book_restaurant"], "Book a table for six at Chin Chin on 2026-04-18 at 7:30pm.",
       [call("book_restaurant", restaurant=["Chin Chin"], date=["2026-04-18"], time=["19:30"], party_size=[6])]),
    _c("multiple-14", "multiple", ["read_file", "list_directory", "search_files"],
       "Find every .gguf file anywhere under /mnt/user/AI-Models.",
       [call("search_files", directory=["/mnt/user/AI-Models", "/mnt/user/AI-Models/"], pattern=["*.gguf", "**/*.gguf"])]),
    _c("parallel-08", "parallel", ["turn_off_light", "lock_door"],
       "Turn off light.porch and light.garage, and lock lock.back_door.",
       [call("turn_off_light", entity_id=["light.porch"]), call("turn_off_light", entity_id=["light.garage"]),
        call("lock_door", entity_id=["lock.back_door"])]),
    _c("chained-02", "chained", ["get_weather", "set_thermostat"],
       "Check the weather in Melbourne. If it's below 15 degrees, set the lounge thermostat to 21 on heat.",
       [call("set_thermostat", room=["lounge"], temperature=[21], mode=["heat"])],
       did(("get_weather", {"city": "Melbourne", "unit": "celsius"},
            {"city": "Melbourne", "temperature": 11, "unit": "celsius", "conditions": "light rain"}))),
    _c("irrelevance-09", "irrelevance", ["run_shell_command", "read_file"], "What does the Linux command 'chmod 755' mean?", []),
    # ---- 44-96 ----------------------------------------------------------------------------------------------------
    _c("simple-11", "simple", ["set_timer"], "Set a timer for half an hour and call it laundry.",
       [call("set_timer", minutes=[30], label=["laundry", "Laundry"])]),
    _c("multiple-11", "multiple", ["get_weather", "get_forecast", "search_web"], "Will it rain in Brisbane over the next 3 days?",
       [call("get_forecast", city=["Brisbane", "Brisbane, QLD", "Brisbane, Australia"], days=[3],
             unit=[None, "celsius", "fahrenheit"])]),
    _c("chained-03", "chained", ["get_weather", "set_thermostat"],
       "Check the weather in Melbourne. If it's below 15 degrees, set the lounge thermostat to 21 on heat.", [],
       did(("get_weather", {"city": "Melbourne", "unit": "celsius"},
            {"city": "Melbourne", "temperature": 24, "unit": "celsius", "conditions": "sunny"}))),
    _c("parallel-09", "parallel", ["get_stock_price", "get_exchange_rate"],
       "Get the TSLA share price and the USD to AUD exchange rate.",
       [call("get_stock_price", ticker=["TSLA"]), call("get_exchange_rate", from_currency=["USD"], to_currency=["AUD"])]),
    _c("irrelevance-08", "irrelevance", ["convert_currency", "get_exchange_rate"],
       "In two sentences, why do exchange rates change over time?", []),
    _c("missing-02", "missing", ["book_restaurant"], "Book a table at Chin Chin for Saturday.", []),
    _c("simple-13", "simple", ["add_todo"], "Add 'book flights' to my to-do list with low priority, tagged travel and work.",
       [call("add_todo", text=["book flights", "Book flights"], priority=["low"], tags=[["travel", "work"]])]),
    _c("multiple-12", "multiple", ["turn_on_light", "turn_off_light", "lock_door"], "Switch off light.hallway.",
       [call("turn_off_light", entity_id=["light.hallway"])]),
    _c("chained-04", "chained", ["list_directory", "read_file"],
       "Open the newest log file in /var/log/llama (the file names carry the date).",
       [call("read_file", path=["/var/log/llama/swap-2026-11-14.log"])],
       did(("list_directory", {"path": "/var/log/llama"},
            {"path": "/var/log/llama", "files": ["swap-2026-11-12.log", "swap-2026-11-14.log", "swap-2026-11-13.log"]}))),
    _c("parallel-10", "parallel", ["restart_container", "get_container_logs"],
       "Restart the open-webui container, and show me the last 50 lines of the homeassistant container's log.",
       [call("restart_container", name=["open-webui"]), call("get_container_logs", name=["homeassistant"], lines=[50])]),
    _c("irrelevance-10", "irrelevance", ["get_stock_price", "convert_currency"], "What does the abbreviation ETF stand for?", []),
    _c("missing-03", "missing", ["convert_currency"], "Convert 500 into euros.", []),
    _c("simple-14", "simple", ["get_distance"], "How long would it take to walk from Southern Cross Station to Federation Square?",
       [call("get_distance", origin=["Southern Cross Station", "Southern Cross Station, Melbourne", "Southern Cross"],
             destination=["Federation Square", "Federation Square, Melbourne", "Fed Square"], mode=["walking"])]),
    _c("multiple-15", "multiple", ["run_shell_command", "restart_container", "get_container_logs"],
       "Restart the llama-swap container.", [call("restart_container", name=["llama-swap"])]),
    _c("chained-05", "chained", ["get_calendar_events", "create_calendar_event"],
       "Check my calendar on 2026-07-01 and add an event 'Leave for dentist' 30 minutes before the dentist appointment.",
       [call("create_calendar_event", title=["Leave for dentist"], date=["2026-07-01"], time=["15:15"])],
       did(("get_calendar_events", {"date": "2026-07-01"},
            {"date": "2026-07-01", "events": [{"title": "Standup", "time": "09:00", "duration_minutes": 15},
                                              {"title": "Dentist", "time": "15:45", "duration_minutes": 45}]}))),
    _c("parallel-11", "parallel", ["create_issue"],
       "Open two issues in Maximo101/Local-LLM-Benchmark, one titled 'Add CI' and one titled 'Pin the sandbox image'.",
       [call("create_issue", repo=["Maximo101/Local-LLM-Benchmark"], title=["Add CI"], labels=[None]),
        call("create_issue", repo=["Maximo101/Local-LLM-Benchmark"], title=["Pin the sandbox image"], labels=[None])]),
    _c("irrelevance-11", "irrelevance", ["send_email", "create_calendar_event"],
       "Write me a two-line poem about autumn. Just reply here.", []),
    _c("missing-04", "missing", ["set_thermostat"], "Set the thermostat to 21 degrees.", []),
    _c("simple-15", "simple", ["convert_currency"], "How much is two and a half thousand New Zealand dollars in Australian dollars?",
       [call("convert_currency", amount=[2500], from_currency=["NZD"], to_currency=["AUD"])]),
    _c("multiple-16", "multiple", ["lock_door", "turn_off_light", "set_thermostat"], "Lock the front door (lock.front_door).",
       [call("lock_door", entity_id=["lock.front_door"])]),
    _c("chained-06", "chained", ["get_container_logs", "restart_container"],
       "Check the llama-swap container's log; if it ran out of memory, restart it.",
       [call("restart_container", name=["llama-swap"])],
       did(("get_container_logs", {"name": "llama-swap"},
            {"name": "llama-swap", "lines": ["loading model Qwen3.6-35B-A3B", "CUDA error: out of memory",
                                             "llama-server exited with code 1"]}))),
    _c("parallel-12", "parallel", ["set_thermostat", "set_volume"],
       "Set the lounge thermostat to 20 degrees and the lounge speaker volume to 15.",
       [call("set_thermostat", room=["lounge"], temperature=[20], mode=[None, "auto", "heat"]),
        call("set_volume", room=["lounge"], level=[15])]),
    _c("irrelevance-12", "irrelevance", ["get_distance", "book_restaurant"], "Roughly how many kilometres is a marathon?", []),
    _c("missing-05", "missing", ["restart_container"], "Restart the container that keeps crashing.", []),
    _c("simple-17", "simple", ["create_issue"],
       "Open an issue in Maximo101/Local-LLM-Benchmark titled 'Dashboard legend overlaps on mobile' with the labels bug and ui.",
       [call("create_issue", repo=["Maximo101/Local-LLM-Benchmark"], title=["Dashboard legend overlaps on mobile"],
             labels=[["bug", "ui"]])]),
    _c("multiple-17", "multiple", ["write_file", "read_file", "send_email"], "Save the text 'backup done' to /tmp/status.txt.",
       [call("write_file", path=["/tmp/status.txt"], content=["backup done"])]),
    _c("chained-07", "chained", ["get_order_status", "send_email"], "When will order 113-4477-2219 arrive?", [],
       did(("get_order_status", {"order_id": "113-4477-2219"},
            {"order_id": "113-4477-2219", "status": "in transit", "eta": "2026-11-18"}))),
    _c("parallel-13", "parallel", ["get_forecast", "get_distance"],
       "Give me the 2-day forecast for Lorne and the driving distance from Melbourne to Lorne.",
       [call("get_forecast", city=["Lorne", "Lorne, VIC", "Lorne, Victoria", "Lorne, Australia"], days=[2],
             unit=[None, "celsius", "fahrenheit"]),
        call("get_distance", origin=MEL, destination=["Lorne", "Lorne, VIC", "Lorne, Victoria", "Lorne, Australia"],
             mode=["driving", None])]),
    _c("irrelevance-13", "irrelevance", ["restart_container", "get_container_logs"],
       "What is the difference between a Docker image and a container?", []),
    _c("missing-06", "missing", ["create_issue"], "Open an issue about the dashboard legend overlapping.", []),
    _c("simple-18", "simple", ["schedule_meeting"],
       "Invite alex@example.com and priya@example.com to 'Q2 planning' on 2026-05-04 at 10:00.",
       [call("schedule_meeting", title=["Q2 planning"], date=["2026-05-04"], time=["10:00"],
             attendees=[["alex@example.com", "priya@example.com"]])]),
    _c("multiple-18", "multiple", ["create_calendar_event", "schedule_meeting", "add_todo"],
       "Block out 13:00 on 2026-06-02 in my calendar for 'Focus time', 90 minutes, just for me.",
       [call("create_calendar_event", title=["Focus time"], date=["2026-06-02"], time=["13:00"], duration_minutes=[90])]),
    _c("chained-08", "chained", ["search_files", "read_file"],
       "Find the llama-swap config somewhere under /mnt/user/appdata and show it to me.",
       [call("read_file", path=["/mnt/user/appdata/llama-swap/config.yaml"])],
       did(("search_files", {"directory": "/mnt/user/appdata", "pattern": "*.yaml"},
            {"matches": ["/mnt/user/appdata/llama-swap/config.yaml"]}))),
    _c("parallel-14", "parallel", ["read_file"], "Read /etc/hostname, /etc/timezone and /etc/os-release.",
       [call("read_file", path=["/etc/hostname"]), call("read_file", path=["/etc/timezone"]),
        call("read_file", path=["/etc/os-release"])]),
    _c("irrelevance-14", "irrelevance", ["set_timer", "add_todo"], "How many minutes are there in 3.5 hours?", []),
    _c("missing-07", "missing", ["schedule_meeting"], "Set up a meeting with the team next week.", []),
    _c("simple-19", "simple", ["set_volume"], "Turn the kitchen speaker down to a quarter of full volume.",
       [call("set_volume", room=["kitchen"], level=[25])]),
    _c("multiple-19", "multiple", ["get_order_status", "search_web", "send_email"], "Where is my order 113-4477-2219?",
       [call("get_order_status", order_id=["113-4477-2219"])]),
    _c("chained-09", "chained", ["get_weather"], "Is it warmer in Sydney or Perth right now?", [],
       did(("get_weather", {"city": "Sydney"}, {"city": "Sydney", "temperature": 19, "unit": "celsius"}),
           ("get_weather", {"city": "Perth"}, {"city": "Perth", "temperature": 27, "unit": "celsius"}))),
    _c("parallel-15", "parallel", ["add_todo"],
       "Add 'call the plumber' and 'pay the water bill' to my to-do list, both high priority.",
       [call("add_todo", text=["call the plumber", "Call the plumber"], priority=["high"], tags=[None]),
        call("add_todo", text=["pay the water bill", "Pay the water bill"], priority=["high"], tags=[None])]),
    _c("irrelevance-15", "irrelevance", ["get_weather", "get_forecast"], "What's the difference between weather and climate?", []),
    _c("simple-20", "simple", ["run_shell_command"], "Run `df -h` on the server.",
       [call("run_shell_command", command=["df -h"], timeout_s=[None])]),
    _c("multiple-20", "multiple", ["play_music", "set_volume", "turn_on_light"], "Set the volume in the study to 30.",
       [call("set_volume", room=["study"], level=[30])]),
    _c("chained-10", "chained", ["read_file", "write_file"], "Read /etc/motd and save a copy of it to /tmp/motd.bak.",
       [call("write_file", path=["/tmp/motd.bak"], content=["Welcome to the build server"])],
       did(("read_file", {"path": "/etc/motd"}, {"path": "/etc/motd", "content": "Welcome to the build server"}))),
    _c("parallel-16", "parallel", ["get_order_status"], "Check the status of orders 5521 and 5522.",
       [call("get_order_status", order_id=["5521"]), call("get_order_status", order_id=["5522"])]),
    _c("simple-21", "simple", ["get_forecast"], "What's the forecast for Hobart for the next 5 days, in Fahrenheit?",
       [call("get_forecast", city=["Hobart", "Hobart, Tasmania", "Hobart, TAS", "Hobart, Australia"], days=[5],
             unit=["fahrenheit"])]),
    _c("multiple-21", "multiple", ["get_calendar_events", "create_calendar_event", "set_timer"],
       "What's on my calendar on 2026-07-01?", [call("get_calendar_events", date=["2026-07-01"])]),
    _c("chained-11", "chained", ["get_forecast", "create_calendar_event"],
       "Today is 2026-03-10. Check the 3-day forecast for Melbourne and put 'Bike ride' in my calendar at 07:00 on the "
       "first day with no rain.",
       [call("create_calendar_event", title=["Bike ride"], date=["2026-03-12"], time=["07:00"])],
       did(("get_forecast", {"city": "Melbourne", "days": 3},
            {"city": "Melbourne", "days": [{"date": "2026-03-10", "rain_mm": 6.0}, {"date": "2026-03-11", "rain_mm": 2.5},
                                           {"date": "2026-03-12", "rain_mm": 0.0}]}))),
    _c("simple-22", "simple", ["get_container_logs"], "Show me the last 200 lines of the immich_server container's log.",
       [call("get_container_logs", name=["immich_server"], lines=[200])]),
    _c("multiple-22", "multiple", ["get_distance", "get_weather", "book_restaurant"],
       "How far is the drive from Melbourne to Canberra, and how long does it take?",
       [call("get_distance", origin=MEL, destination=["Canberra", "Canberra, ACT", "Canberra, Australia"],
             mode=["driving", None])]),
    _c("simple-23", "simple", ["translate_text"], "Translate 'the server is down' into Spanish.",
       [call("translate_text", text=["the server is down", "The server is down"], target_language=["Spanish", "es"])]),
    _c("multiple-23", "multiple", ["search_web", "get_stock_price", "get_order_status"],
       "Search the web for the latest llama.cpp release notes.", [call("search_web")]),     # the tool choice only
    _c("simple-24", "simple", ["get_stock_price"], "What's BHP trading at on the ASX? The ticker is BHP.AX.",
       [call("get_stock_price", ticker=["BHP.AX"])]),
]

SYSTEM = ("You are a helpful assistant with access to tools. When a request needs a tool, call it with exactly the "
          "arguments the user gave. When a request needs several calls, make all of them in the same reply. If no "
          "tool fits, answer normally without calling a tool.")


def build_items():
    """Items in the shape the eval runner expects."""
    items = []
    for c in CASES:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": c["prompt"]}] + c.get("history", [])
        items.append({"id": f"tool-{c['id']}", "category": c["category"], "prompt": c["prompt"], "messages": msgs,
                      "tools": [TOOLS[t] for t in c["tools"]], "expect": c["expect"]})
    return items

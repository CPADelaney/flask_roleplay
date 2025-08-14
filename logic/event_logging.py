async def log_event(user_id, conversation_id, event_type, data):
    from logic.game_time_helper import GameTimeContext
    async with GameTimeContext(user_id, conversation_id) as game_time:
        event = {
            "event_type": event_type,
            "timestamp": await game_time.to_datetime(),
            "game_time": {
                "year": game_time.year,
                "month": game_time.month,
                "day": game_time.day,
                "time_of_day": game_time.time_of_day,
            },
            "time_string": await game_time.to_string(),
            **data,
        }
    return event

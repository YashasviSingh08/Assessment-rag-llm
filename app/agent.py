def detect_intent(query: str):

    q = query.lower()

    appointment_keywords = [
        "appointment",
        "book",
        "schedule",
        "doctor",
        "visit",
        "consult"
    ]

    for keyword in appointment_keywords:

        if keyword in q:
            return "appointment"

    return "medical"
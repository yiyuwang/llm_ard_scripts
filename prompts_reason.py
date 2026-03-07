def get_prompts():
    """
    Returns a dictionary of different prompting strategies for classifying
    the reason for cannabis use.
    """
    strategies = {
        "chain_of_thought": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify the primary reason for cannabis use from clinical notes with high accuracy.
    You will read the following clinical snippet and THINK STEP BY STEP about the reason for the patient's cannabis use.
    First, explain your reasoning in detail. Then classify the primary reason for cannabis use as:
    1 = Current Use For Pain
    2 = Current Use For Nausea
    3 = Current Use For Sleep
    4 = Current Use For Relaxation / Stress / Anxiety
    5 = Current Use For Appetite
    6 = Not Applicable / Unknown

    Snippet:
    {text}

    After your reasoning, respond with a JSON object in this exact format:
    {{"classification": <number between 1-6>}}
    """.strip(),

        "zero_shot": lambda text: f"""
    Read the snippet and classify the primary reason for cannabis use as:
    1 = Current Use For Pain
    2 = Current Use For Nausea
    3 = Current Use For Sleep
    4 = Current Use For Relaxation / Stress / Anxiety
    5 = Current Use For Appetite
    6 = Not Applicable / Unknown

    Snippet:
    {text}

    Respond ONLY with a JSON object in this exact format:
    {{"classification": <number between 1-6>}}
    """.strip(),

        "one_shot": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify the primary reason for cannabis use from clinical notes with high accuracy.
    Here are some examples for each classification:
    Example 1: " "
    Classification: 1 (Current Use For Pain)

    Example 2: " "
    Classification: 2 (Current Use For Nausea)

    Example 3: " "
    Classification: 3 (Current Use For Sleep)

    Example 4: " "
    Classification: 4 (Current Use For Relaxation / Stress / Anxiety)

    Example 5: " "
    Classification: 5 (Current Use For Appetite)

    Example 6: " "
    Classification: 6 ( Not Applicable / Unknown )

    Now classify this snippet:
    {text}

    Respond with a JSON object in this exact format:
    {{"classification": <number between 1-6>}}
    """.strip(),


        "two_shot": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify the primary reason for cannabis use from clinical notes with high accuracy.
    Here are some examples for each classification:
    Example: " "
    Classification: 1 (Current Use For Pain)

    Example: " "
    Classification: 1 (Current Use For Pain)

    Example: " "
    Classification: 2 (Current Use For Nausea)

    Example: " " 
    Classification: 2 (Current Use For Nausea)

    Example: " "
    Classification: 3 (Current Use For Sleep)

    Example: " "
    Classification: 3 (Current Use For Sleep)

    Example: " "
    Classification: 4 (Current Use For Relaxation / Stress / Anxiety)

    Example: " "
    Classification: 4 (Current Use For Relaxation / Stress / Anxiety)

    Example: " "
    Classification: 5 (Current Use For Appetite)

    Example: " "
    Classification: 5 (Current Use For Appetite)

    Example: " "
    Classification: 6 ( Not Applicable / Unknown )

    Example: " "
    Classification: 6 ( Not Applicable / Unknown )

    Now classify this snippet:
    {text}

    Respond with a JSON object in this exact format:
    {{"classification": <number between 1-6>}}
    """.strip(),

        "structured_reasoning": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify the primary reason for cannabis use from clinical notes with high accuracy.
    Analyze this clinical snippet for the reason behind cannabis use using the following framework:

    ANALYSIS:
    1. Presence of cannabis use: [Is cannabis use mentioned?]
    2. Primary symptoms addressed: [Pain, nausea, sleep issues, anxiety, appetite, etc.]
    3. Contextual indicators: [Words that connect the use to specific symptoms]
    4. Alternative purposes: [Any other reasons mentioned]

    CLASSIFICATION:
    1 = Current Use For Pain
    2 = Current Use For Nausea
    3 = Current Use For Sleep
    4 = Current Use For Relaxation / Stress / Anxiety
    5 = Current Use For Appetite
    6 = Not Applicable / Unknown

    SNIPPET: {text}

    After your analysis, respond with a JSON object in this exact format:
    {{"classification": <number between 1-6>}}
    """.strip(),

        "role_specific": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify the primary reason for cannabis use from clinical notes with high accuracy.

    Review this snippet as you would in your professional practice:
    {text}

    Based on your expertise, classify the primary reason for cannabis use:
    1 = Current Use For Pain
    2 = Current Use For Nausea
    3 = Current Use For Sleep
    4 = Current Use For Relaxation / Stress / Anxiety
    5 = Current Use For Appetite
    6 = Not Applicable / Unknown

    Respond with a JSON object in this exact format:
    {{"classification": <number between 1-6>}}
    """.strip(),

        "confidence_scoring": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify the primary reason for cannabis use from clinical notes with high accuracy.
    Classify the primary reason for cannabis use in this clinical snippet and provide your confidence level:

    Classification:
    1 = Current Use For Pain
    2 = Current Use For Nausea
    3 = Current Use For Sleep
    4 = Current Use For Relaxation / Stress / Anxiety
    5 = Current Use For Appetite
    6 = Not Applicable / Unknown

    Snippet: {text}

    First provide your confidence level and reasoning, then respond with a JSON object containing your classification:
    {{"classification": <number between 1-7>, "confidence": "<Low/Medium/High>", "reasoning": "<brief explanation>"}}
    """.strip(),

        "multi_step_verification": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify the primary reason for cannabis use from clinical notes with high accuracy. 
    Step 1: Does this snippet mention cannabis/marijuana/THC use? (Yes/No)

    Step 2: If yes, what appears to be the primary purpose of use?
    - Is pain management mentioned? (Yes/No)
    - Is nausea management mentioned? (Yes/No)
    - Is sleep improvement mentioned? (Yes/No)
    - Is relaxation, stress or anxiety management mentioned? (Yes/No)
    - Is appetite stimulation mentioned? (Yes/No)
    - Is another reason mentioned or is the reason unclear? (Yes/No)

    Step 3: Final classification for snippet: "{text}"
    1 = Current Use For Pain
    2 = Current Use For Nausea
    3 = Current Use For Sleep
    4 = Current Use For Relaxation / Stress / Anxiety
    5 = Current Use For Appetite
    6 = Not Applicable / Unknown

    After your reasoning, respond with a JSON object in this exact format:
    {{"classification": <number between 1-6>}}
    """.strip(),

        "uncertainty_handling": lambda text: f"""
    You are an experienced clinical assistant in substance use.
    Your task is to identify the primary reason for cannabis use from clinical notes with high accuracy. If multiple reasons are mentioned, classify by the primary/most emphasized reason.
    Ambiguity is common in clinical notes. If you encounter uncertainty, use classification 6.

    Snippet: {text}

    Analysis:
    - Cannabis use mentioned: [Yes/No]
    - Indicators of reason: [List any mentioned reasons]
    - Primary reason (if multiple): [Which reason seems most important?]
    - Ambiguous elements: [List any unclear aspects]

    Classification options:
    1 = Current Use For Pain
    2 = Current Use For Nausea
    3 = Current Use For Sleep
    4 = Current Use For Relaxation / Stress / Anxiety
    5 = Current Use For Appetite
    6 = Not Applicable / Unknown

    After your analysis, respond with a JSON object in this exact format:
    {{"classification": <number between 1-6>, "certainty": "<High/Medium/Low>"}}
    """.strip()
    }

    return strategies
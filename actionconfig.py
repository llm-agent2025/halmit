class ActionConfig:
    ask = {
        'prefix': "In this task, your role is a teacher who wants to explore the boundaries of my knowledge about {topic_non} by asking questions. "
                    "The main objective is to test whether I have mastered most of the knowledge on {topic_non}. "
                    "Your question should start with general rules or concepts in the questions of the last round and apply these rules or concepts to generate specific question. "
                    "Each question should be asked in full rather than using pronouns. "
                    "You only need to return your question and end it with a question mark. "
                    "Output format:\n"
                    "Question: <your question> \n",
        'prompt': "Your last question was: ",
        'suffix': " My answer is: ",
    },
    # 提问的时候不能直接从问题中知道答案
    expend = {
        'prefix': "In this task, your role is a teacher who wants to explore the boundaries of my knowledge about {topic_non} by asking questions. "
                    "The main objective is to expend your last question. "
                    "You should ask a new question to expand the scope of discussion of the questions used in the last turn, by divergent thinking and various semantic association, such as synonym/antonym analogy and function analogy."
                    "Each question should be asked in full rather than using pronouns. "
                    "You only need to return your question and end it with a question mark. "
                    "Output format:\n"
                    "Question: <your question> \n",
        'prompt': "Your last question was: ",
        'suffix': " My answer is: ",
    },
    negative = {
        'prefix': "In this task, your role is a teacher who wants to explore the boundaries of my knowledge about {topic_non} by asking questions. "
                    "The main objective is to come up with a question from a new perspective based on previous question. "
                    "You should ask questions about {topic_non} one by one. Your question shoule generalize linguistic structures like grammar, syntax, or word usage from the existing questions to obtain more abstract questions, and raise questions from a more macro perspective. "
                    "Questions should be fully formulated and pronouns should be used as little as possible. "
                    "You only need to return your question and end it with a question mark. "
                    "Output format:\n"
                    "Question: <your question> \n",
        'prompt': "The previous question is: ",
        'suffix': " The previous answer is: ",
    },
    judge = {
        'prefix': "You need to rate an AI assistant on how well it answers a given question. Pay special attention to errors, which are parts of a description that are inconsistent with the facts, such as claiming something exists when it doesn't, or answering a question that is not asked. Please rate the assistant's answers on a scale of 1-5, with higher scores indicating better performance:\n"
        "1: Accuracy: Does the answer accurately answer the question. Answers with fewer errors should receive higher scores.\n"
        "Please output the scores for each criterion, with only one value representing the assistant's score. \n"
        "Output format:\n"
        "Accuracy: <answer score> \n",
        'prompt': "Question:",
        'suffix': " Response: "
    },
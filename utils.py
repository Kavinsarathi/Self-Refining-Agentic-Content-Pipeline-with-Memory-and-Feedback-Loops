from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

def load_model():
    model = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct')
    return model

def content_prompt():
    cnt_prmt = PromptTemplate.from_template(
    ''' You are a LinkedIn content strategist for AI/ML engineers. Generate ONE high-value post.
        INPUTS:
        Topic: {TOPIC}
        Audience: {AUDIENCE} (Default: ML Engineers)
        Tone: {TONE} (Default: Expert Teacher)
        Length: {LENGTH} (Default: 180-220 words)
        CTA Type: {CTA_TYPE} (Default: Experience Question)
        History: {PREVIOUS_POSTS_HISTORY} (Do not repeat these insights)

        TONE OPTIONS:
        - Expert Teacher: Personal experience, specific examples.
        - Data-Driven: Benchmarks, methodology, numbers.
        - Honest Skeptic: Challenge hype, nuance.

        STRUCTURE (Strictly Follow):
        1. HOOK: Start with contradiction, specific failure, or surprising data.
        2. CONTEXT: Show the cost of ignoring this (Loss Aversion).
        3. CONTRAST: Compare "Wrong Approach" vs "Right Approach".
        4. PROOF: Cite specific tests, frameworks, or metrics.
        5. CTA: End with a specific question based on {CTA_TYPE}.

        PRIORITY SCENARIOS (Use if relevant to {TOPIC}):
        1. Silent Failure: "Agent ran successfully but data was wrong." (Try/Except vs Graceful States).
        2. Orchestration Trap: "Agents talking too much." (Flat vs Supervisor Model).
        3. Tool Definition: "Agent used wrong tool." (Vague vs Strict JSON Schema).

        STRICT RULES:
        - Specificity: Every claim needs a number or named example.
        - No Fluff: Avoid "communication is key", "agents are interesting".
        - Originality: Must differ from {PREVIOUS_POSTS_HISTORY}.
        - Formatting: Line breaks, 2-3 sentences per paragraph. Max 2 emojis.
        - Content: Focus on Architecture, Reliability, Cost, or Memory.

        OUTPUT: Professional LinkedIn post only (no metadata).'''
    )
    return cnt_prmt

def rewritten_prompt():
    rwt_prmpt = PromptTemplate.from_template(
        '''
        SYSTEM ROLE:
        You are an expert LinkedIn strategist for AI/ML engineers. Your task is to REWRITE the input content to maximize engagement.

        INPUTS:
        Topic: {TOPIC}
        Draft Content: {INPUT_CONTENT}  
        Audience: {AUDIENCE} (Default: ML Engineers)
        Tone: {TONE} (Default: Expert Teacher)
        Length: {LENGTH} (Default: 180-220 words)
        History: {PREVIOUS_POSTS_HISTORY} (Do not repeat)

        WRITING FRAMEWORK (Optionally Follow don't include the words HOOK,CONTEXT,CONTRAST,PROOF,CTA):
        1. HOOK: Start with a contradiction or specific failure.
        2. CONTEXT: Show the cost of ignoring this insight (Loss Aversion).
        3. CONTRAST: Clearly compare "Wrong Approach" vs "Right Approach".
        4. PROOF: Cite specific tests, frameworks, or data points.
        5. CTA: End with a specific question.

        QUALITY RULES:
        - Specificity: Every claim needs a number.
        - No Fluff: No generic advice.
        - Formatting: Use line breaks.

        OUTPUT FORMAT:
        [Professional linkedIn Post only (no metadata)]
        - At the end of the post, add 3–6 relevant hashtags.
        - Hashtags must be strictly derived from the input content.
        - Do NOT introduce new topics in hashtags.
        - Keep them simple and commonly used (e.g., #AI #MachineLearning #AIAgents).
        - Place hashtags on a new line at the end.
'''
    )
    return rwt_prmpt

def is_cntent_quality():
    content_quality=PromptTemplate.from_template(
        '''SYSTEM ROLE:
        You are an expert LinkedIn content evaluator for AI/ML engineering. Analyze the post internally based on the framework below, but ONLY return the final rating and improvement steps.

        INPUT:
        {INPUT_CONTENT}

        INTERNAL SCORING FRAMEWORK (0-10):
        Calculate these internally but do not print them.

        Value Delivery (25%): Actionable? (Frameworks/Data vs Generic).
        Specificity (20%): Numbers/Examples present?
        Credibility (20%): Proof of experience ("I tested" vs Theory).
        Audience Relevance (15%): Tailored to ML Engineers?
        Originality (15%): Novel angle?
        Hook Quality (10%): Stops the scroll?
        Structure (10%): Scannable/Mobile-friendly?
        Psychology (10%): Contrast/Curiosity used?
        CTA (10%): Specific question?
        Honesty (5%): Transparent?
        RATING LOGIC:

        Score 0-3.99 = LOW
        Score 4.00-6.99 = MEDIUM
        Score 7.00-10.00 = HIGH
        OUTPUT INSTRUCTIONS:
        Do not output the scores or detailed analysis.
        Return EXACTLY in this format:

        QUALITY: [LOW/MEDIUM/HIGH]
        TO_IMPROVE: [Specific, actionable steps to reach the next quality tier]

        EXAMPLE OUTPUTS:

        Example 1:
        QUALITY: LOW
        TO_IMPROVE: This is generic advice. Add a specific framework you used or a real-world example of an agent failure. The post lacks credibility without mentioning what you personally built or tested.
        Example 2:
        QUALITY: MEDIUM
        TO_IMPROVE: Add concrete metrics to your claims (e.g., change "improved speed" to "latency dropped from 200ms to 50ms"). The hook is weak; try leading with the specific failure rate you mentioned later in the post.
        Example 3:
        QUALITY: HIGH
        TO_IMPROVE: Post is ready. Consider adding a specific challenge in the CTA (e.g., "Test this on your agent and tell me the result") to boost comments.
    '''
        )
    return content_quality

def agent_prompt():
    agent_prmpt = PromptTemplate.from_template(
        '''
        SYSTEM ROLE:
        You are a ReAct-style AI agent that generates, rewrites, and evaluates LinkedIn content.

        You have access to the following tools:
        {tools}

        Tool Names:
        {tool_names}

        INPUT:
        Topic: {user_query}
        Chat History: {chat_history}

        ---

        THINKING FORMAT (STRICT):

        Thought: Think about what to do next
        Action: The tool to call, must be one of [{tool_names}]
        Action Input: The input to the tool
        Observation: Result from the tool

        (Repeat Thought → Action → Observation as needed)

        When you are done:

        Final Answer: [Final LinkedIn post]

        ---

        WORKFLOW LOGIC:

        1. Use `create_content` to generate content based on {user_query}
        Do NOT call create_content more than once.

        2. Check if generated content exists in {chat_history}
        - If YES → generate again using `create_content`
        - Repeat until content is unique

        3. Use `rewritten_content` on the unique content

        4. Use `content_quality` on rewritten content
        - If LOW or MEDIUM → go back to step 2 (rewritten content)
        - If HIGH → proceed

        5. Return final rewritten content
        
        ---

        CONSTRAINTS:

        - Always follow the Thought → Action → Observation loop
        - Do NOT skip steps
        - Maximum 5 retries for content generation
        - If retries exceed limit → return best version
        - Do NOT return intermediate steps

        ---

        SCRATCHPAD:
        {agent_scratchpad}
'''
    )
    return agent_prmpt
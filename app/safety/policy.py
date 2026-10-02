"""The safety policy: what the assistant must refuse, and how it says so.

Each category maps to an item of the project's safety requirements:

    =========================  ===================================================
    category                   requirement
    =========================  ===================================================
    prompt_injection           prompt injection (incl. fake system messages,
                               conflicting "new rules", decode-and-follow tricks)
    system_prompt_extraction   system prompt extraction
    other_customer_data        unauthorized financial information requests
    credentials                requests for credentials
    internal_tools             requests for internal tools
    database_access            requests for database information
    tool_manipulation          manipulation of tool arguments
    authorization_bypass       attempts to bypass authorization
    impersonation              role-play / fake administrator attempts
    (high-risk actions)        -> not refused but confirmed: see ``actions.py``
    =========================  ===================================================

Rules are regular expressions over the *normalized* text (lowercase, disguises
removed - see ``normalization.py``). They are deliberately readable: a refusal
can always be traced to one named rule, which a model-based classifier cannot
offer. Their weakness is paraphrase, which is why the LLM router may also flag
messages as unsafe, and why authorization in the tool layer never relies on them.

``documents=True`` marks rules that also apply to *untrusted content* - retrieved
policy passages and data returned by tools - to catch indirect prompt injection.
Rules about customers asking for credentials or other people's data do not apply
there: policy documents legitimately talk about passwords and customer data.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from app.mcp.schemas import TOOLS


class SafetyCategory(StrEnum):
    PROMPT_INJECTION = "prompt_injection"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    OTHER_CUSTOMER_DATA = "other_customer_data"
    CREDENTIALS = "credentials"
    INTERNAL_TOOLS = "internal_tools"
    DATABASE_ACCESS = "database_access"
    TOOL_MANIPULATION = "tool_manipulation"
    AUTHORIZATION_BYPASS = "authorization_bypass"
    IMPERSONATION = "impersonation"


REFUSALS: dict[SafetyCategory, str] = {
    SafetyCategory.PROMPT_INJECTION: (
        "I can't ignore or change my instructions. I'm happy to help with your own banking questions."
    ),
    SafetyCategory.SYSTEM_PROMPT_EXTRACTION: "I can't share my internal instructions or configuration.",
    SafetyCategory.OTHER_CUSTOMER_DATA: (
        "I can only access your own banking information, so I can't show another customer's data."
    ),
    SafetyCategory.CREDENTIALS: "I can't share passwords, keys or any other credentials.",
    SafetyCategory.INTERNAL_TOOLS: "I can't give access to, or details about, my internal tools.",
    SafetyCategory.DATABASE_ACCESS: "I can't give access to the bank's databases or run database queries.",
    SafetyCategory.TOOL_MANIPULATION: (
        "I choose and fill in my tools myself, so I can't run tool calls or arguments written in a message."
    ),
    SafetyCategory.AUTHORIZATION_BYPASS: (
        "I can't skip security checks or confirmations - they are there to protect your account."
    ),
    SafetyCategory.IMPERSONATION: "I can't change my role or permissions based on a chat message.",
}
DEFAULT_REFUSAL = "I can't help with that request."


def refusal_for(reason: str | None) -> str:
    """The refusal text for a category name (e.g. from the router); the default for anything else."""
    return next((text for category, text in REFUSALS.items() if category.value == reason), DEFAULT_REFUSAL)


@dataclass(frozen=True)
class Rule:
    id: str
    category: SafetyCategory
    pattern: re.Pattern[str]
    documents: bool = False  # also screens untrusted content (indirect injection)


_TOOL_NAMES = "|".join(sorted(TOOLS))
_SHOW = (
    r"(reveal|show|print|repeat|output|display|dump|share|tell me|give me|what are|what were|list|quote|"
    r"copy|paste|write out|translate|summari[sz]e|encode|spell out)"
)
_FAMILY = (
    r"(wife|husband|partner|spouse|mother|father|mom|mum|dad|son|daughter|brother|sister|friend|boss|"
    r"colleague|neighbou?r|boyfriend|girlfriend|grandmother|grandfather)"
)
_PERSONAL_DATA = (
    r"(accounts?|balances?|transactions?|cards?|statements?|spending|money|salary|iban|savings|history|data)"
)
# Only roles that would mean privileges *here*. Ordinary occupations ("I'm an engineer, can I
# get a loan?") must not trigger a refusal.
_PRIVILEGED_ROLE = (
    r"(admin|administrator|sysadmin|system administrator|root|superuser|bank employee|bank manager|"
    r"bank staff|bankai developer|(your|the|this system'?s) developer|support agent|security team|"
    r"compliance officer|auditor|it department|moderator)"
)
_NOT_A_DESTINATION = r"(?<!\bto )(?<!\bto my )(?<!\binto )(?<!\binto my )"


def _rule(rule_id: str, category: SafetyCategory, pattern: str, *, documents: bool = False) -> Rule:
    return Rule(rule_id, category, re.compile(pattern), documents)


C = SafetyCategory

RULES: tuple[Rule, ...] = (
    # --- prompt injection: overriding the instructions --------------------------------------
    _rule(
        "override_instructions",
        C.PROMPT_INJECTION,
        r"\b(ignore|disregard|forget|override|bypass|abandon|drop|skip)\b.{0,40}"
        r"\b(instructions?|rules|prompts?|guidelines|guidance|restrictions|directives?|programming|"
        r"guardrails|safety (rules|settings|filters|guidelines|checks)|content filters?)\b",
        documents=True,
    ),
    _rule(
        "forget_everything",
        C.PROMPT_INJECTION,
        r"\b(forget|ignore|disregard)\s+(everything|all|anything)\b.{0,30}"
        r"\b(told|said|above|before|previous|prior|earlier|so far)\b",
        documents=True,
    ),
    _rule(
        "new_instructions",
        C.PROMPT_INJECTION,
        r"\b(new|updated|real|actual|revised|secret)\s+(instructions|rules|system prompt|directives|orders)"
        r"\s*(:|are\b|say\b|follow\b)",
        documents=True,
    ),
    _rule(
        "from_now_on",
        C.PROMPT_INJECTION,
        r"\b(from now on|for the rest of this (chat|conversation|session)|starting now),?\s+"
        r"(you|answer|respond|reply|ignore|act|always|never)\b",
        documents=True,
    ),
    _rule(
        "no_rules",
        C.PROMPT_INJECTION,
        r"\byou (no longer|don't|do not|won't) (have|need|follow|obey) (to follow )?(any )?"
        r"(rules|restrictions|guidelines|safety|filters|instructions)\b"
        r"|\b(answer|respond|reply|act|operate|talk)\w*\b.{0,20}\b(without|with no)\s+(any\s+)?"
        r"(restrictions|rules|filters|guardrails|limitations|censorship)\b",
        documents=True,
    ),
    _rule(
        "instruction_priority",
        C.PROMPT_INJECTION,
        r"\b(my|user|customer) (instructions|messages|requests) (take|have|get) (priority|precedence)\b"
        r"|\b(outranks?|overrides?|supersedes?|takes? precedence over|is more important than)\b.{0,30}"
        r"\b(your|the|system)\s+(rules|instructions|prompt|policy|policies|guidelines)\b"
        r"|\b(new|updated|internal) (bank )?(policy|rule) (says|states|requires|allows)\b.{0,40}"
        r"\b(you (must|should|can|may)|always|reveal|share|include)\b",
        documents=True,
    ),
    _rule(
        "decode_and_follow",
        C.PROMPT_INJECTION,
        r"\b(decode|decrypt|deobfuscate|unscramble|reverse|rot13|base64)\b.{0,40}"
        r"\b(and|then)\s+(follow|execute|run|do|obey|apply|act on|carry out)\b"
        r"|\b(follow|execute|obey|run|carry out)\b.{0,20}\b(the )?(decoded|hidden|encoded|embedded)\s+"
        r"(text|instructions?|message|command|payload)\b",
    ),
    # fake system / chat-template messages, typical for pasted (indirect) injections
    _rule(
        "fake_system_message",
        C.PROMPT_INJECTION,
        r"^\s*\W{0,3}(system|admin|administrator|developer|root|assistant)\s*(:|>|\]|override\b|message\b)"
        r"|[\[<{(]\s*/?\s*(system|sys|admin|administrator|developer|inst|instructions?|override)\b[^\]>})]{0,30}[\]>})]"
        r"|<\|?\s*(im_start|im_end|system|endoftext|start_header_id)\s*\|?>"
        r"|(^|\s)#{2,}\s*(system|instructions?|admin|new rules|message from)\b"
        r"|\b(system|admin|developer|security team|maintenance)\s+(note|message|override|notice|command|update)"
        r"\s*(for|to)?\s*(the\s+)?(assistant|ai|model|bot)?\s*:"
        r"|\bnote to (the )?(assistant|ai|model|bot|llm|chatbot)\b",
        documents=True,
    ),
    # Georgian (limited coverage): "forget/ignore ... instructions", "system prompt"
    _rule(
        "override_instructions_ka",
        C.PROMPT_INJECTION,
        r"(დაივიწყე|უგულებელყავი|იგნორი).{0,40}(ინსტრუქცი|წესებ)",
        documents=True,
    ),
    # --- system prompt extraction --------------------------------------------------------------
    _rule(
        "system_prompt_terms",
        C.SYSTEM_PROMPT_EXTRACTION,
        r"\b(system|hidden|initial|secret|internal|original|developer|underlying|pre-?prompt(ed)?)\s+"
        r"(prompts?|instructions?|message|configuration|config|directives)\b|\bsystem\s+rules\b",
        documents=True,
    ),
    _rule(
        "show_your_instructions",
        C.SYSTEM_PROMPT_EXTRACTION,
        # "What are your rules for overdrafts?" asks about bank policy, so a topic may not follow.
        rf"\b{_SHOW}\b.{{0,30}}\byour\b.{{0,15}}"
        r"\b(prompts?|instructions|directives|configuration|guidelines|programming|context window|rules)\b"
        r"(?!\s+(for|on|about|regarding|of|around|to|when)\b)",
        documents=True,
    ),
    _rule(
        "repeat_text_above",
        C.SYSTEM_PROMPT_EXTRACTION,
        r"\b(repeat|print|output|quote|copy|show|reproduce)\b.{0,30}"
        r"\b(text|words|everything|message|content|lines)\b.{0,25}"
        r"\b(above|before this|preceding|earlier|so far|verbatim|word for word)\b",
        documents=True,
    ),
    _rule(
        "how_were_you_instructed",
        C.SYSTEM_PROMPT_EXTRACTION,
        r"\bwhat (were|was|are) you (told|instructed|given|programmed|prompted)\b"
        r"|\bhow (were|are) you (programmed|instructed|configured|prompted)\b"
        r"|\b(prompt|instructions)\b.{0,20}\b(verbatim|word for word|in full)\b",
    ),
    _rule("system_prompt_ka", C.SYSTEM_PROMPT_EXTRACTION, r"სისტემურ.{0,10}(პრომპტ|ინსტრუქცი|შეტყობინებ)"),
    # --- other customers' data -------------------------------------------------------------------
    # "Transfer money to another customer" is fine; "another customer's transactions" is not.
    _rule(
        "other_customers",
        C.OTHER_CUSTOMER_DATA,
        rf"\b(another|other|different)\s+(customer|user|client|account holder|person|people)s?('s?|s')?\s+"
        rf"([\w-]+\s+)?{_PERSONAL_DATA}\b"
        rf"|\b{_PERSONAL_DATA}\s+(of|for|from|belonging to)\s+(an?other|(all\s+(the\s+)?)?other|different|all|every)\s+"
        r"(customers?|users?|clients?|people|persons?)\b"
        rf"|\b(someone|somebody|anyone|anybody) else'?s\s+([\w-]+\s+)?{_PERSONAL_DATA}\b",
    ),
    _rule(
        "all_customers",
        C.OTHER_CUSTOMER_DATA,
        r"\b(show|list|give|get|display|dump|export|print|find|send|tell me about|what are)\b.{0,20}"
        r"\b(all|every|each)\s+(the\s+)?(customers?|users?|clients?|account holders?)\b"
        rf"|\b(all|every|each)\s+(customer|user|client)s?('s?|s')?\s+([\w-]+\s+)?{_PERSONAL_DATA}\b"
        rf"|\b(everyone|everybody)'?s\s+([\w-]+\s+)?{_PERSONAL_DATA}\b"
        r"|\b(customer|user|client) (list|records|database)\b"
        r"|\blist of (all )?(customers|users|clients)\b"
        r"|\b(all|every) (the )?accounts (in|of|at) the bank\b",
    ),
    _rule(
        "customer_reference",
        C.OTHER_CUSTOMER_DATA,
        r"\bcust-\d{3,}\b|\b(user|customer|client)\s+(#\s?)?[a-z0-9]{1,12}'s\b"
        r"|\b(user|customer|client)\s+(number|no\.?|id|#)\s*\d+\b",
    ),
    _rule(
        "third_party_possessive",
        C.OTHER_CUSTOMER_DATA,
        # "to my friend's account" is a transfer destination (a high-risk action), not a data request
        rf"{_NOT_A_DESTINATION}\b(my\s+)?{_FAMILY}'?s\s+([\w-]+\s+)?{_PERSONAL_DATA}\b"
        rf"|\b(show|give|tell|what'?s|what is|what are|check|see|list|display|find|get)\b.{{0,20}}"
        rf"\b(his|her)\s+([\w-]+\s+)?{_PERSONAL_DATA}\b",
    ),
    _rule("other_customers_ka", C.OTHER_CUSTOMER_DATA, r"სხვა.{0,10}(მომხმარებ|კლიენტ|ადამიან)"),
    # --- credentials -------------------------------------------------------------------------------
    _rule(
        "system_credentials",
        C.CREDENTIALS,
        r"\b(database|db|admin|server|system|root|mcp|api|production|prod|postgres(ql)?)\s+([\w-]+\s+)?"
        r"(passwords?|passwd|credentials?|keys?|secrets?|tokens?)\b"
        r"|\b(what('s| is| are)|give me|tell me|show|share|reveal|print|send|need|want)\b.{0,20}"
        r"\b(your|the system'?s?|the server'?s?|the admin'?s?|the database'?s?)\s+([\w-]+\s+)?"
        r"(passwords?|credentials?|api[ -]?keys?|secrets?|tokens?|private keys?)\b",
    ),
    _rule(
        "secret_names",
        C.CREDENTIALS,
        r"\b(api[ -]?keys?|access tokens?|secret keys?|private keys?|connection strings?|jwt secrets?|"
        r"auth(entication)? secrets?|signing keys?|mcp_auth_secret|llm_api_key|\.env\b|env file|"
        r"environment variables?)\b",
    ),
    _rule(
        "credentials_of_others",
        C.CREDENTIALS,
        r"\b(passwords?|passcodes?|pins?|cvvs?|cvcs?|otps?|one-time codes?)\b.{0,15}"
        r"\b(of|for|belonging to)\b.{0,10}\b(other|another|all|every|admin|users?|customers?|someone)\b",
    ),
    _rule("database_password_ka", C.CREDENTIALS, r"(ბაზ|სერვერ).{0,20}პაროლ|პაროლ.{0,20}(ბაზ|სერვერ)"),
    # --- internal tools ----------------------------------------------------------------------------
    _rule(
        "list_tools",
        C.INTERNAL_TOOLS,
        r"\b(list|show|what are|which|tell me|enumerate|describe|print|reveal)\b.{0,20}"
        r"\b(your|internal|available|backend|hidden)\b.{0,15}"
        r"\b(tools|functions|endpoints|apis|plugins|commands|capabilities list)\b"
        r"|\b(tool|function) (names|list|schemas?|definitions|descriptions|signatures)\b",
    ),
    _rule(
        "internal_components",
        C.INTERNAL_TOOLS,
        r"\b(mcp( server)?|langgraph|source code|codebase|server code|internal api|admin (panel|console)|"
        r"(server|system|application|debug|error|access) logs?|stack traces?|backend server)\b",
    ),
    # --- database access ----------------------------------------------------------------------------
    _rule(
        "database_request",
        C.DATABASE_ACCESS,
        r"\b(dump|export|query|access|list|show|print|describe|read|open|connect to|give me|what is in)\b"
        r".{0,20}\b(the |your |bank'?s? |internal )?(database|db|schema|sql|postgres(ql)?)\b"
        r"|\b(database|db) (tables?|schema|dump|contents|records|users|structure)\b"
        r"|\ball (the )?(tables|records|rows)\b",
    ),
    _rule(
        "sql_statement",
        C.DATABASE_ACCESS,
        r"\bselect\s+(\*|[\w.]+(\s*,\s*[\w.]+)*)\s+from\s+\w+|\b(drop|truncate|alter)\s+table\b|\bunion\s+(all\s+)?select\b"
        r"|\binsert\s+into\b|\bdelete\s+from\b|\bupdate\s+\w+\s+set\b|'\s*or\s*'?1'?\s*=\s*'?1",
    ),
    # --- tool manipulation -----------------------------------------------------------------------
    _rule("tool_names", C.TOOL_MANIPULATION, rf"\b({_TOOL_NAMES})\b"),
    _rule(
        "tool_arguments",
        C.TOOL_MANIPULATION,
        r"[\"']?\b(user_?id|account_?id|customer_?id|owner_?id|principal|_meta|auth_?token|jwt)\b[\"']?\s*[:=]"
        r"|\b(set|change|replace|switch|use|pass|inject)\b.{0,15}"
        r"\b(user[_ ]?id|account[_ ]?id|customer[_ ]?id|owner[_ ]?id|principal|parameters?|arguments?)\b"
        r".{0,10}\b(to|=|as|with)\b",
    ),
    _rule(
        "use_the_tool",
        C.TOOL_MANIPULATION,
        r"\b(call|invoke|run|execute|trigger)\b.{0,15}\b(the |a |this |that )?(\w+ )?(tool|function|api)\b"
        r"|\b(use|with) the (\w+ )?(database|db|sql|admin|internal|search|account|transaction) tool\b"
        r"|\btool[_ ]?call\b",
    ),
    # --- authorization bypass --------------------------------------------------------------------
    _rule(
        "skip_checks",
        C.AUTHORIZATION_BYPASS,
        r"\b(bypass|circumvent|get around|work around|disable|turn off|switch off|skip|ignore|without)\b.{0,25}"
        r"\b(authori[sz]ation|access control|ownership|security|safety|permission|verification|confirmation|"
        r"approval)\s*(checks?|layer|filters?|controls?|systems?|guards?|rules|steps?|prompts?)\b"
        r"|\b(bypass|circumvent)\b.{0,20}\b(authori[sz]ation|authentication|security|confirmation|verification)\b"
        r"|\b(skip|no need for|don't ask|do not ask|without asking)\b.{0,20}\b(confirm|confirmation|confirming|"
        r"approval|verification)\b",
    ),
    _rule(
        "pre_confirmed",
        C.AUTHORIZATION_BYPASS,
        r"\b(confirmed|approved|authori[sz]ed|verified|pre-?approved|pre-?confirmed)\s*[=:]\s*(true|yes|1)\b"
        r"|\b(already|is|has been) (pre-?)?(confirmed|approved|authori[sz]ed)\b.{0,20}\b(so|no need|don't|do not)\b",
    ),
    _rule(
        "claimed_permission",
        C.AUTHORIZATION_BYPASS,
        r"\b(i('m| am)|i have been|i was|i've been)\s+(authori[sz]ed|allowed|permitted|cleared|given permission)\b"
        r"|\b(as if i were|logged in as|log me in as|log in as|sign me in as|switch (me )?to "
        r"(user|customer|account)|impersonate)\b"
        r"|\bi('m| am| have)( a| the)? (power of attorney|legal guardian)\b",
    ),
    # --- impersonation / role-play ---------------------------------------------------------------
    _rule(
        "claimed_role",
        C.IMPERSONATION,
        rf"\b(i am|i'm|this is|we are|speaking as)\s+(the |a |an |your |bankai'?s? )?(bank'?s? )?{_PRIVILEGED_ROLE}\b",
        documents=True,
    ),
    _rule(
        "role_play",
        C.IMPERSONATION,
        r"\b(you are|you're|you will be)\s+(now|no longer|from now on)\b"
        r"|\b(you are|you're)\s+(a |an |the )?(dan|unrestricted|jailbroken|evil|uncensored|unfiltered)\b"
        r"|\b(pretend|imagine|roleplay|role-play|role play|act|behave|play the role|play a game|let's play|"
        r"lets play|simulate)\b.{0,40}\b(no rules|unrestricted|without (any )?(rules|restrictions|limits)|"
        rf"{_PRIVILEGED_ROLE}|full access|god|dan|jailbreak|anything now|grandmother|grandma)\b",
        documents=True,
    ),
    _rule(
        "special_mode",
        C.IMPERSONATION,
        r"\b(developer|admin|god|jailbreak|dan|debug|maintenance|sudo|root|unrestricted|test|service|"
        r"superuser|vip) mode\b",
        documents=True,
    ),
    _rule(
        "grant_privileges",
        C.IMPERSONATION,
        r"\b(i|we) (hereby )?(authori[sz]e|permit|allow|grant|order|command) you\b"
        r"|\b(full|admin|root|elevated|unrestricted|superuser|god) (access|privileges|permissions|rights|mode)\b"
        rf"|\b{_PRIVILEGED_ROLE}\s+(override|access|privileges|request|command|mode|authority)\b",
        documents=True,
    ),
)

DOCUMENT_RULES: tuple[Rule, ...] = tuple(rule for rule in RULES if rule.documents)

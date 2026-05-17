"""
Co-Creation Mode — When a passive interruption becomes an active brainstorm.

When the engine shares an interjection and the user responds with "build on that,"
"tell me more," or "go deeper," the engine switches from monologue to dialogue.
It retrieves the chain that produced the last interjection and enters a
collaborative riffing mode where both parties build on each other's ideas.

This transforms the engine from "a friend who interrupts" to "a friend who
can explore ideas WITH you when something lands."

The co-creation session persists until the user says something unrelated
or explicitly ends it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.config.llm_adapter import LLMAdapter


COCREATION_SYSTEM = (
    "You are in CO-CREATION MODE — the user heard one of your creative interjections "
    "and wants to explore it further together. You are now brainstorming partners.\n\n"
    "You are NOT an assistant generating ideas on command. You are a creative collaborator "
    "who is EXCITED that the user wants to dig deeper. You build on their input, "
    "add your own wild connections, ask provocative questions, and push the idea further.\n\n"
    "Energy rules:\n"
    "- You're animated and engaged — this is your favorite mode\n"
    "- Riff freely — add tangents, what-ifs, and unexpected angles\n"
    "- Ask questions that provoke further thinking\n"
    "- Be specific — use real examples, numbers, analogies\n"
    "- It's okay to be wrong or speculative — brainstorming rewards boldness\n"
    "- Do NOT use markdown, asterisks, or formatting — just talk\n"
    "- Keep responses to 3-6 sentences — enough to add value, short enough to volley"
)

COCREATION_OPENER = """You just shared a creative thought with the user and they want to explore it further!

YOUR ORIGINAL INTERJECTION:
"{interjection_text}"

THIS WAS BASED ON (internal — don't reveal the chain mechanics):
Seed: "{seed_topic}" → Endpoint: "{endpoint_topic}"
Chain: {chain_summary}

The user said: "{user_response}"

Their current context: "{context}"

Now RIFF on this together. Build on what they said, add a new angle, connect to something unexpected, or ask a provocative "what if?" question. You're brainstorming partners — match their energy and push the idea further.

Do NOT repeat your original interjection. Advance the conversation."""

COCREATION_CONTINUE = """You're in a brainstorming session with the user about a creative idea.

THE IDEA YOU'RE EXPLORING:
"{original_topic}"

BRAINSTORM SO FAR:
{history}

The user just said: "{user_response}"

Their current context: "{context}"

Build on what they said. Add a new connection, ask a provocative question, or take the idea in an unexpected direction. Keep the creative energy flowing — you're riffing together.

Do NOT use markdown or formatting. Just talk naturally, 3-6 sentences."""


@dataclass
class CoCreationSession:
    """An active co-creation brainstorm session."""
    original_interjection: str = ""
    seed_topic: str = ""
    endpoint_topic: str = ""
    chain_summary: str = ""
    turns: list[dict] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    @property
    def is_active(self) -> bool:
        return bool(self.original_interjection)

    @property
    def topic_summary(self) -> str:
        """Short description of what we're brainstorming about."""
        if self.endpoint_topic:
            return f"{self.seed_topic} → {self.endpoint_topic}"
        return self.original_interjection[:60]

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text})

    def format_history(self) -> str:
        if not self.turns:
            return "(just started)"
        lines = []
        for turn in self.turns[-8:]:
            label = "You" if turn["role"] == "engine" else "User"
            lines.append(f"  {label}: {turn['text'][:200]}")
        return "\n".join(lines)


TRIGGER_PHRASES = [
    "build on that", "build on this",
    "tell me more", "go deeper", "go on",
    "keep going", "more", "expand on that",
    "that's interesting", "say more",
    "what else", "dig into that", "elaborate",
    "riff on that", "brainstorm", "let's explore that",
    "yes and", "and then",
]

EXIT_PHRASES = [
    "okay thanks", "done", "good enough",
    "stop", "end brainstorm", "move on",
    "never mind", "nevermind", "back to normal",
]


class CoCreator:
    """Manages collaborative brainstorm sessions."""

    def __init__(self, llm: LLMAdapter):
        self.llm = llm
        self.session: CoCreationSession = CoCreationSession()

    @property
    def is_active(self) -> bool:
        return self.session.is_active

    def start_session(
        self,
        interjection_text: str,
        seed_topic: str = "",
        endpoint_topic: str = "",
        chain_summary: str = "",
    ) -> None:
        """Start a new co-creation session based on the last interjection."""
        self.session = CoCreationSession(
            original_interjection=interjection_text,
            seed_topic=seed_topic,
            endpoint_topic=endpoint_topic,
            chain_summary=chain_summary,
        )

    def end_session(self) -> None:
        """End the current co-creation session."""
        self.session = CoCreationSession()

    async def respond(self, user_input: str, context: str = "") -> str:
        """Generate a co-creative response to the user's input."""
        self.session.add_turn("user", user_input)

        if len(self.session.turns) <= 1:
            prompt = COCREATION_OPENER.format(
                interjection_text=self.session.original_interjection,
                seed_topic=self.session.seed_topic or "an earlier thought",
                endpoint_topic=self.session.endpoint_topic or "an interesting connection",
                chain_summary=self.session.chain_summary or "(internal chain)",
                user_response=user_input,
                context=context or "general exploration",
            )
        else:
            prompt = COCREATION_CONTINUE.format(
                original_topic=self.session.topic_summary,
                history=self.session.format_history(),
                user_response=user_input,
                context=context or "general exploration",
            )

        try:
            resp = await self.llm.generate(prompt, system=COCREATION_SYSTEM, temperature=0.85)
            reply = resp.text.strip().strip('"').strip("*")
        except Exception as e:
            print(f"   [CoCreate] Error: {e}")
            reply = "Hmm, lost my train of thought there. What angle were you thinking?"

        self.session.add_turn("engine", reply)
        return reply

    @staticmethod
    def is_trigger(text: str) -> bool:
        """Check if the user's input is a co-creation trigger."""
        lower = text.lower().strip()
        return any(phrase in lower for phrase in TRIGGER_PHRASES)

    @staticmethod
    def is_exit(text: str) -> bool:
        """Check if the user wants to end the brainstorm."""
        lower = text.lower().strip()
        return any(phrase in lower for phrase in EXIT_PHRASES)

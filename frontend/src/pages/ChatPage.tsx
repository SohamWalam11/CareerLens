import { FormEvent, useState } from "react";

type Message = {
  role: "assistant" | "user";
  content: string;
};

const starterMessages: Message[] = [
  {
    role: "assistant",
    content: "Hello Alex! I can explain why roles are recommended and how to close your skill gaps."
  }
];

const ChatPage = () => {
  const [messages, setMessages] = useState<Message[]>(starterMessages);
  const [input, setInput] = useState("");

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!input.trim()) return;

    const newMessages: Message[] = [
      ...messages,
      { role: "user", content: input.trim() },
      {
        role: "assistant",
        content:
          "Thanks for the question! I will soon reference curated insights from the CareerLens knowledge graph."
      }
    ];

    setMessages(newMessages);
    setInput("");
  };

  return (
    <section className="flex h-full flex-col space-y-6">
      <header>
        <h2 className="text-3xl font-semibold text-neon-blue">AI Career Coach</h2>
        <p className="mt-2 text-slate-400">
          Ask CareerLens about recommended paths, required skills, or next steps to accelerate your
          journey.
        </p>
      </header>

      <div className="flex-1 overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60 shadow">
        <div className="h-full overflow-y-auto p-6 space-y-4">
          {messages.map((message: Message, index: number) => (
            <div
              key={`${message.role}-${index}`}
              className={`max-w-2xl rounded-lg px-4 py-3 text-sm leading-relaxed ${
                message.role === "assistant"
                  ? "bg-slate-800 text-slate-100"
                  : "ml-auto bg-neon-blue/20 text-neon-blue"
              }`}
            >
              <p>{message.content}</p>
            </div>
          ))}
        </div>
      </div>

      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-3 rounded-full border border-slate-800 bg-slate-900/70 px-4 py-2"
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask how to close the data storytelling gap..."
          className="flex-1 bg-transparent text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none"
        />
        <button
          type="submit"
          className="rounded-full bg-neon-pink px-5 py-2 text-sm font-semibold text-slate-950 hover:bg-neon-purple"
        >
          Send
        </button>
      </form>
    </section>
  );
};

export default ChatPage;

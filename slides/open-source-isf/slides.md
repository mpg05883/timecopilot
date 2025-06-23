---
title: LLMs meet foundation time series models, are we ready for forecasting agents?
author: Azul Garza (azul.garza.r@gmail.com, @AzulGarza)
patat:
    incrementalLists: true
    slideLevel: 1
    margins:
        left: 5
        right: 5
    images:
        backend: auto
...

> Hi 👋

- i'm Azul
- i'm also a proud mexican trans woman 🇲🇽🏳️‍⚧️
- today i'm going to speak about some ideas around forecasting agents

---

> what's all this hype around ai agents?

- agents are trending across domains 
    - code development
    - research
    - chat and productivity
    - web and browsing
    - enterprise workflows
    - multimodal assistants
- searches for "AI agents" have surged dramatically


---

![ai-agents](img/ai-agent.png)

---

> agents are *everywhere*

- GitHub Copilot agents
- GPTs as autonomous task-runners
- ReAct, AutoGPT, BabyAGI, Devin

---

> and the agent wave will grow

---

> but what *is* an AI agent?


---

![ai-agents-def](img/ai-agent-def.png)

---

> agent runtime

- **orchestration:** what’s the goal? what are the rules?
- **memory:** 
    - short-term: what's happening right now?
    - long-term: what do we already know about the user?

- **reasoning & planning:** thinks through what to do next

---

> two planning modes:

- without feedback
    - chain of thought: break tasks into step-by-step reasoning  
    - tree of thought: explore multiple branches of reasoning  

- with feedback
    - react: reason + act + observe  
    - reflexion: learn from past mistakes  
    - human-in-the-loop: refine with external feedback  
    - useful for open-ended, adaptive tasks

---

> model

- the brain. usually a large language model doing the heavy thinking

---

> tools

- external helpers: functions, apis, or models the agent can call to get things done  
  (for example: forecast(), plot(), explain_model())

---

> a simpler definition, by LangChain

- "a system that uses an LLM to decide the control flow of an application"

---

> a mathematical intuition

```
loop:
  observe environment
  reason (LLM)
  choose best action
  update state
  repeat until goal or termination
```

- agents = planning + tool use + feedback

---

> why now? LLMs changed everything

- LLMs are:
  - general reasoners
  - tool orchestrators
  - planners with memory
- foundation models made agents practical, not just theoretical

---

> agents across modalities


---

![multimodality](img/multimodal-agents.png)


---

> where are the forecasting agents?

- forecasting = high-impact, technical, widely needed
- but: little agent-based experimentation
- yet it’s a perfect fit:
  - model selection
  - error diagnosis
  - data cleaning
  - code generation

---

> opportunity: open source can be a good place to experiment

- current focus on language, vision, robotics
- time series is often underhyped
- we have the tools:
  - foundation time series models, open source tools (Nixtla, GluonTS, SkTime, etc...)
  - LangChain, OpenAI, Claude, Mistral

---

> why forecasting needs agents, and why now

- forecasting is complex and messy
- agents make it navigable and human-friendly
- no one’s cracked this yet

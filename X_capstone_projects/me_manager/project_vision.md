# MeManager: Project Vision & Architecture

## Overview
MeManager is an autonomous, multi-agent personal development and time management system. It goes beyond simple to-do lists by actively orchestrating long-term goals (e.g., "Become a Senior Python Developer"). It breaks these granular goals into dynamic, daily actionable chunks, assesses user progress, and visually presents a cohesive daily plan across multiple independent pursuits.

## Core Mechanics

### 1. Goal Definition & Initialization
*   A User inputs a high-level goal and a daily time constraint (e.g., "I want to learn Machine Learning, 1 hour a day").
*   A dedicated **Goal Agent** is spawned exclusively for this pursuit.
*   **Assessment Phase:** The Agent may optionally quiz the user or ask for current skills to establish a baseline starting point.
*   **Roadmap Generation:** Using web search (Serper or custom tools), the Agent constructs a detailed, sequential roadmap bridging the gap between current skills and the end goal.

### 2. The Daily Drip (Execution)
*   The system is strictly state-based, not calendar-based. Progress only advances when the work is actually done.
*   **The Daily Set:** The user logs in and requests "the next set of things to do today."
*   **Content Mix:** A daily set is a curated mix of:
    *   *New Material* (Reading, coding exercises)
    *   *Assessments* (Quizzes to prove mastery of yesterday's material)
    *   *Reinforcement* (Spaced repetition of older concepts)
*   **Pacing:** If a user misses a day, the set remains unchanged. If a user finishes early and wants to get ahead, they can request the explicit "next set."

## Multi-Agent Architecture

MeManager relies on a localized society of specialized agents. Because each goal can be wildly different, isolation of concerns is critical.

### 1. The Goal Agents (The Workers)
*   **Role:** One agent per active tracking goal (e.g., PythonAgent, FitnessAgent, PortugueseAgent).
*   **Responsibility:** Only cares about its assigned domain. Manages its own RAG state, breaks down its roadmap, assesses the user's domain knowledge, and generates the exact tasks for the *Daily Set* related to its goal.
*   **Memory:** Needs strong episodic memory to remember what the user struggled with 3 weeks ago to reinforce it today.

### 2. The Visualizer Agent (The Architect/Designer)
*   **Role:** The timeline and presentation asset generator.
*   **Responsibility:** Ingests the current state and overall roadmaps from *all* active Goal Agents.
*   **Capabilities:** Leverages advanced multimodal models (e.g., Grok Imagine, Veo, Imagen, or similar vision/video generation APIs) alongside code generation (Mermaid.js, HTML/CSS).
*   **Output:** Creates highly engaging, rich visual assets. This includes dynamic timelines, motivational images, beautiful charts mapping progress, and polished presentation slides designed to keep the user engaged.

### 3. The Summarizer / Presenter Agent (The Manager)
*   **Role:** The user-facing interface.
*   **Responsibility:** Wakes up when the user asks for their daily tasks. It queries all Goal Agents for today's required workload, takes the chart from the Visualizer, and formats a cohesive, encouraging, and clear "Daily Presentation" for the user.

## Data Flow & State Management
1.  **User Request:** "Give me my day."
2.  **Orchestration:** Presenter Agent pings Goal Agent A (Python) and Goal Agent B (Fitness).
3.  **Task Generation:** Goal Agents review user history, build today's chunk, and pass structured JSON tasks back to the Presenter.
4.  **Visualization:** Presenter passes the raw timelines to Visualizer Agent, retrieving a generated chart.
5.  **Delivery:** Presenter compiles the Dashboard (Tasks + Chart) and presents it to the User.

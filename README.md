# Restro AI Agent

A conversational restaurant ordering agent built with [LangGraph](https://langchain-ai.github.io/langgraph/) and [Groq](https://groq.com/). Customers chat with **Max**, the restaurant assistant, who takes orders, checks the menu, "cooks" and "serves" the food, and hands back a bill — all orchestrated as a state-machine graph.

## How it works

The agent is a LangGraph `StateGraph` with four nodes. Each order flows through them, with conditional routing on success/failure:

```
customer input
      │
   chatbot ──(off-topic / chit-chat)── END
      │ (order placed & confirmed)
      ▼
  take_order ──(not on menu / out of stock)── chatbot ── END
      │ (approved; stock reserved)
      ▼
    cook ──(kitchen failure)── chatbot ── END
      │ (cooked)
      ▼
   server ──(delivery fails, retry < 2)── cook   (retry loop)
      │                └(retry ≥ 2)── chatbot ── END
      │ (served)
      ▼
   bill → END
```

| Node | Responsibility |
|------|----------------|
| **chatbot** | Talks to the customer, detects orders via a bound `place_order` tool, confirms before placing, and relays any failures back. Restricted to restaurant topics only. |
| **take_order** | Validates the dish against the menu, reserves stock, and auto-adjusts the quantity if fewer are available than requested. |
| **cook** | Prepares the food (simulated 90% success rate). |
| **server** | Delivers the order and generates the itemized bill (simulated 85% success; failures retry via `cook` up to twice). |

### Features

- **Order confirmation** — Max confirms the dish and price before placing the order (`yes`/`no`).
- **Live stock tracking** — each menu item has a quantity that decreases as orders are filled; stock is reserved once at approval so retries never double-count.
- **Graceful fallbacks** — unavailable items, kitchen failures, and delivery failures are all explained to the customer.
- **Topic guardrail** — Max politely declines off-topic questions and steers back to the menu.

## Tech stack

- **LangGraph** — graph/state-machine orchestration
- **langchain-groq** — LLM via Groq (`openai/gpt-oss-20b`)
- **python-dotenv** — loads the API key from `.env`

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/lazywolf01/restro-ai-agent.git
cd restro-ai-agent
python -m venv venv
```

Activate it:

```bash
# Windows (PowerShell)
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Copy the example env file and add your key (get one free at [console.groq.com](https://console.groq.com/)):

```bash
cp .env.example .env
```

Then edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

```bash
python Restro_agent.py
```

Then chat in the terminal:

```
You: Hi!
Max: Hello! How can I help you today?

You: Can I order 2 burgers?
Max: You'd like 2x Burger ($12.99 each). Shall I place the order? (yes/no)

You: yes
Max: Order placed for 2x Burger! We'll serve you soon.
Max:
=============================================
  Your order is here! Enjoy your meal!
=============================================
  2x Burger               $  25.98
  Tax (10%)                $   2.60
  -----------------------------------
  TOTAL                    $  28.58
=============================================
```

Type `quit` to exit.

## Menu

| Dish | Price | Stock |
|------|-------|-------|
| Burger | $12.99 | 5 |
| Pizza | $15.99 | 3 |
| Pasta | $10.99 | 7 |
| Salad | $8.99 | 10 |
| Steak | $25.99 | 2 |
| Soup | $6.99 | 8 |
| Sandwich | $9.99 | 6 |
| Ice Cream | $4.99 | 15 |

Edit the `MENU` dictionary in `Restro_agent.py` to change dishes, prices, or stock.

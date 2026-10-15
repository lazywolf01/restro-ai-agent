import os
import re
import random
from typing import TypedDict, Annotated, Literal
import operator

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# ── Menu ─────────────────────────────────────────────────────────────────────
MENU: dict[str, dict] = {
    "burger":    {"price": 12.99, "quantity": 5},
    "pizza":     {"price": 15.99, "quantity": 3},
    "pasta":     {"price": 10.99, "quantity": 7},
    "salad":     {"price":  8.99, "quantity": 10},
    "steak":     {"price": 25.99, "quantity": 2},
    "soup":      {"price":  6.99, "quantity": 8},
    "sandwich":  {"price":  9.99, "quantity": 6},
    "ice cream": {"price":  4.99, "quantity": 15},
}

# ── State ─────────────────────────────────────────────────────────────────────
class RestaurantState(TypedDict):
    messages: Annotated[list, operator.add]
    order_item: str        # dish the customer wants
    order_qty: int         # how many
    order_status: str      # pending | order_placed | approved | cooked | served
                           # | unavailable | cook_failed | serve_failed
    bill: float
    server_retries: int    # how many times server has tried to deliver

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

SYSTEM_PROMPT = """\
You are Max, a friendly assistant for a restaurant. You ONLY handle restaurant
matters: the menu, dishes, prices, and taking food orders.

When a customer wants to order food, call the `place_order` tool with the dish
name and quantity. For greetings or questions about the menu, reply with text
and do NOT call the tool.

If the customer asks about anything unrelated to the restaurant or food,
politely decline — explain you can only help with the menu and orders — and ask
whether they'd like to order something. Keep it short and warm.

Available menu items: {menu_items}
"""


class place_order(BaseModel):
    """Place a food order for a customer who wants to order a dish."""
    item: str = Field(description="The dish the customer wants to order")
    quantity: int = Field(default=1, description="How many of the dish")


llm_with_tools = llm.bind_tools([place_order])

# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Chatbot
# Handles customer conversation and relays kitchen/server failures back.
# ─────────────────────────────────────────────────────────────────────────────
AFFIRM = {"yes", "yeah", "yep", "yup", "ya", "sure", "ok", "okay", "confirm",
          "confirmed", "correct", "please", "y"}
DENY   = {"no", "nope", "nah", "cancel", "stop", "dont", "n"}


def chatbot_node(state: RestaurantState) -> dict:
    status = state.get("order_status", "pending")

    # Customer is replying to an order confirmation (yes / no)
    if status == "awaiting_confirmation":
        words = set(re.findall(r"[a-z]+", state["messages"][-1].content.lower()))
        item, qty = state["order_item"], state["order_qty"]
        if words & AFFIRM:
            msg = f"Order placed for {qty}x {item.title()}! We'll serve you soon."
            return {"messages": [AIMessage(content=msg)], "order_status": "order_placed"}
        if words & DENY:
            msg = "No problem, I've cancelled that. Would you like to order something else?"
            return {"messages": [AIMessage(content=msg)], "order_status": "pending"}
        msg = f"Sorry, I didn't catch that — shall I place the order for {qty}x {item.title()}? (yes/no)"
        return {"messages": [AIMessage(content=msg)], "order_status": "awaiting_confirmation"}

    # Relay failure from take_order
    if status == "unavailable":
        available = {k: v for k, v in MENU.items() if v["quantity"] > 0}
        lines = "\n".join(
            f"  • {k.title():<15} ${v['price']:.2f}  ({v['quantity']} left)"
            for k, v in available.items()
        )
        msg = (
            f"Sorry, '{state['order_item']}' isn't available right now.\n"
            f"Here's what we currently have:\n{lines}\n\nWhat would you like to order?"
        )
        return {"messages": [AIMessage(content=msg)], "order_status": "pending"}

    # Relay failure from cook
    if status == "cook_failed":
        msg = (
            f"I'm so sorry! Our kitchen ran into an issue preparing your "
            f"'{state['order_item']}'. Would you like to try something else?"
        )
        return {"messages": [AIMessage(content=msg)], "order_status": "pending"}

    # Relay failure from server (too many retries)
    if status == "serve_failed":
        msg = (
            f"We sincerely apologize — we're having trouble serving your order "
            f"right now. Please let us know if you'd like to try again."
        )
        return {"messages": [AIMessage(content=msg)], "order_status": "pending"}

    # Normal conversation: send to LLM with the order tool bound
    menu_items = ", ".join(MENU.keys())
    system = SYSTEM_PROMPT.format(menu_items=menu_items)
    full_messages = [SystemMessage(content=system)] + state["messages"]

    response = llm_with_tools.invoke(full_messages)

    # If the model called place_order, confirm with the customer first
    if response.tool_calls:
        args = response.tool_calls[0]["args"]
        item = str(args.get("item", "")).lower().strip()
        qty = max(1, int(args.get("quantity", 1)))
        if item in MENU:
            msg = (f"You'd like {qty}x {item.title()} (${MENU[item]['price']:.2f} each). "
                   f"Shall I place the order? (yes/no)")
        else:
            msg = f"You'd like {qty}x {item.title()}. Shall I place the order? (yes/no)"
        return {
            "messages": [AIMessage(content=msg)],
            "order_item": item,
            "order_qty": qty,
            "order_status": "awaiting_confirmation",
            "server_retries": 0,
        }

    # No order — just chat
    return {"messages": [AIMessage(content=response.content.strip())], "order_status": "pending"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Take Order
# Validates against menu. Approves or rejects.
# ─────────────────────────────────────────────────────────────────────────────
def take_order_node(state: RestaurantState) -> dict:
    item = state["order_item"]
    qty  = state["order_qty"]
    print(f"\n[TAKE ORDER] Checking menu for: {qty}x '{item}'")

    if item not in MENU:
        print(f"[TAKE ORDER] '{item}' not on menu.")
        return {"order_status": "unavailable"}

    available_qty = MENU[item]["quantity"]

    if available_qty == 0:
        print(f"[TAKE ORDER] '{item}' is out of stock.")
        return {"order_status": "unavailable"}

    if available_qty < qty:
        print(f"[TAKE ORDER] Only {available_qty} left — adjusting order quantity.")
        MENU[item]["quantity"] -= available_qty
        notice = (
            f"Quick note: we only have {available_qty}x {item.title()} left, "
            f"so I've adjusted your order from {qty} to {available_qty}."
        )
        return {
            "order_qty": available_qty,
            "order_status": "approved",
            "messages": [AIMessage(content=notice)],
        }

    MENU[item]["quantity"] -= qty
    print(f"[TAKE ORDER] Order approved. Remaining stock of '{item}': {MENU[item]['quantity']}")
    return {"order_status": "approved"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Cook
# Prepares the food. 90 % success rate.
# ─────────────────────────────────────────────────────────────────────────────
def cook_node(state: RestaurantState) -> dict:
    item = state["order_item"]
    qty  = state["order_qty"]
    print(f"\n[COOK] Preparing {qty}x '{item}'...")

    success = random.random() > 0.10   # 90 % success

    if success:
        print(f"[COOK] Done! {qty}x '{item}' ready to serve.")
        return {"order_status": "cooked"}
    else:
        print(f"[COOK] Kitchen issue — could not prepare '{item}'.")
        return {"order_status": "cook_failed"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Server
# Delivers the food and generates bill. 85 % success; retries via cook.
# ─────────────────────────────────────────────────────────────────────────────
def server_node(state: RestaurantState) -> dict:
    item    = state["order_item"]
    qty     = state["order_qty"]
    retries = state.get("server_retries", 0)
    print(f"\n[SERVER] Delivering {qty}x '{item}' (attempt {retries + 1})...")

    success = random.random() > 0.15   # 85 % success

    if success:
        unit_price  = MENU[item]["price"]
        subtotal    = unit_price * qty
        tax         = subtotal * 0.10
        grand_total = subtotal + tax

        bill_text = (
            f"\n{'='*45}\n"
            f"  Your order is here! Enjoy your meal!\n"
            f"{'='*45}\n"
            f"  {qty}x {item.title():<20} ${subtotal:>7.2f}\n"
            f"  {'Tax (10%)':<24} ${tax:>7.2f}\n"
            f"  {'-'*35}\n"
            f"  {'TOTAL':<24} ${grand_total:>7.2f}\n"
            f"{'='*45}\n"
            f"  Thank you for dining with us!\n"
            f"{'='*45}"
        )
        print(f"[SERVER] Served! Bill: ${grand_total:.2f}")
        return {
            "order_status": "served",
            "bill": grand_total,
            "messages": [AIMessage(content=bill_text)],
        }
    else:
        print(f"[SERVER] Could not deliver. Sending back to kitchen.")
        return {
            "order_status": "serve_failed",
            "server_retries": retries + 1,
        }


# ── Routing ───────────────────────────────────────────────────────────────────
def route_chatbot(state: RestaurantState) -> Literal["take_order", "__end__"]:
    return "take_order" if state.get("order_status") == "order_placed" else "__end__"

def route_take_order(state: RestaurantState) -> Literal["cook", "chatbot"]:
    return "cook" if state["order_status"] == "approved" else "chatbot"

def route_cook(state: RestaurantState) -> Literal["server", "chatbot"]:
    return "server" if state["order_status"] == "cooked" else "chatbot"

def route_server(state: RestaurantState) -> Literal["cook", "chatbot", "__end__"]:
    if state["order_status"] == "serve_failed":
        if state.get("server_retries", 0) < 2:
            print(f"[ROUTER] Server failed — retrying via cook (retry #{state['server_retries']}).")
            return "cook"
        else:
            print(f"[ROUTER] Server failed too many times — notifying customer.")
            return "chatbot"
    return "__end__"


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(RestaurantState)

    g.add_node("chatbot",    chatbot_node)
    g.add_node("take_order", take_order_node)
    g.add_node("cook",       cook_node)
    g.add_node("server",     server_node)

    g.set_entry_point("chatbot")

    g.add_conditional_edges("chatbot",    route_chatbot,     {"take_order": "take_order", "__end__": END})
    g.add_conditional_edges("take_order", route_take_order,  {"cook": "cook", "chatbot": "chatbot"})
    g.add_conditional_edges("cook",       route_cook,        {"server": "server", "chatbot": "chatbot"})
    g.add_conditional_edges("server",     route_server,      {"cook": "cook", "chatbot": "chatbot", "__end__": END})

    return g.compile()


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    app = build_graph()

    menu_display = "\n".join(
        f"  • {k.title():<15} ${v['price']:.2f}   ({v['quantity']} available)"
        for k, v in MENU.items()
    )
    print("\n" + "=" * 50)
    print("      Welcome to AI Restaurant!")
    print("=" * 50)
    print(f"\nToday's Menu:\n{menu_display}\n")
    print("Type 'quit' to exit.\n")

    # Persistent state across turns
    state: RestaurantState = {
        "messages":      [],
        "order_item":    "",
        "order_qty":     0,
        "order_status":  "pending",
        "bill":          0.0,
        "server_retries": 0,
    }

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nThank you for visiting AI Restaurant. Goodbye!")
            break

        # Add user message and run one graph turn
        prev_len = len(state["messages"])
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        result = app.invoke(state)

        # Merge result into persistent state
        state.update(result)

        # Keep "awaiting_confirmation" alive across turns; otherwise reset
        if state["order_status"] != "awaiting_confirmation":
            state["order_status"]   = "pending"
            state["server_retries"] = 0

        # Print every new AI message produced this turn
        for msg in result["messages"][prev_len + 1:]:
            if isinstance(msg, AIMessage):
                print(f"\nMax: {msg.content}\n")


if __name__ == "__main__":
    run()

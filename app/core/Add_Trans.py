"""
Smart Finance Assistant Chatbot
- Extracts structured transactions from user input with an LLM
- Categorizes and inserts into a Supabase database
- Provides feedback and validation
"""

# 1. Imports
import os
import re
from dotenv import load_dotenv
from langchain_fireworks import Fireworks
from datetime import datetime, timezone
from dateparser import parse as date_parse
from supabase import create_client, Client

# 2. ENV Setup
load_dotenv()
API_KEY = os.getenv("API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 3. LLM & DB Initialization
llm = Fireworks(api_key=API_KEY, model="accounts/fireworks/models/deepseek-v3", temperature=1.0, max_tokens=1024)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 4. Prompt for Transaction Extraction

def process_transaction_with_llm(text):
    transaction_prompt = """
    Extract transaction details from the following text and classify the amount as either income or expense (TYPE).
    Extract date (handle relative dates like 'yesterday'), a description, and the category.
    If a specific date is not mentioned, assume today's date. If you can't classify the amount as either income or expense,
    assume it is expense and keep it uncategorized. Provide a natural response about the spending.

    Text: {input_text}

    Current date: {current_date}

    Format your response as follows:
    CREATED_AT: [Extract the date, handle relative dates like 'this morning', 'yesterday', 'last week', etc. ISO8601 date, e.g., 2025-06-16T15:30:00Z],
    AMOUNT: [numeric amount],
    TYPE: [Income or Expense],
    CATEGORY: [Classify into one of if expense: Food, Health, Transportation, Entertainment, Fashion, Lifestyle, Education
               Classify into one of if income: Salary, Freelance, Investments, Bonus, Refunds]
    DESCRIPTION: [one or two word description of the transaction]
    FEEDBACK: [Provide a natural, conversational response about the spending. Consider:
    - If it's a good deal or expensive for that category
    - Suggest money-saving tips if relevant
    - Compliment good financial decisions
    - Express concern for unusually high spending
    - Comment on the timing or necessity of the purchase
    Make it sound natural and varied.]

    Example:
    Input: "I got today 2,000 from upwork"
    CREATED_AT: 2024-01-30T00:00:00Z
    AMOUNT: 2000
    TYPE: Income
    CATEGORY: Freelance
    DESCRIPTION: upwork
    FEEDBACK: That’s wonderful—congrats on your Freelance job!

    Input: "Bought lunch for 50 yesterday"
    CREATED_AT: 2024-01-29T00:00:00Z
    AMOUNT: 50
    TYPE: Expense
    CATEGORY: Food
    DESCRIPTION: lunch
    FEEDBACK: That's a reasonable amount for lunch! If you're looking to save more, you might consider bringing lunch from home occasionally.
    """

    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00')
    formatted_prompt = transaction_prompt.format(
        input_text=text,
        current_date=current_date
    )
    # Call LLM
    response = llm.invoke(formatted_prompt)
    return response

# 5. LLM Output Parser

def parse_llm_response(response):
    """Parse the LLM's output to get transaction fields."""
    # Extract CREATED_AT (or fallback to now)
    now = datetime.now(timezone.utc)
    date_match = re.search(r"CREATED_AT:\s*([^\n]+)", response)
    date_str = date_match.group(1).strip() if date_match else None
    if not date_str or date_str.lower().startswith("unknown"):
        date_obj = now
    else:
        date_obj = date_parse(date_str)
        if not date_obj:
            date_obj = now
        elif date_obj.tzinfo is None:
            date_obj = date_obj.replace(tzinfo=timezone.utc)
        if date_obj > now:
            return {"error": "❌ Cannot add transaction: date is in the future."}
    created_at = date_obj.strftime('%Y-%m-%d %H:%M:%S+00')

    # Extract AMOUNT
    amount_match = re.search(r"AMOUNT:\s*([\d.]+)", response, re.IGNORECASE)
    amount = float(amount_match.group(1)) if amount_match else 0.0

    # Extract TYPE
    type_match = re.search(r"TYPE:\s*([^\n]+)", response, re.IGNORECASE)
    transaction_type = type_match.group(1).strip().rstrip(",. ").capitalize() if type_match else None

    # Extract CATEGORY
    category_match = re.search(r"CATEGORY:\s*([^\n]+)", response)
    category = category_match.group(1).strip() if category_match and category_match.group(1).strip() else None

    # Extract DESCRIPTION
    desc_match = re.search(r"DESCRIPTION:\s*([^\n]+)", response, re.IGNORECASE)
    description = desc_match.group(1).strip() if desc_match and desc_match.group(1).strip() else None

    # Extract FEEDBACK
    feedback_match = re.search(r"FEEDBACK:\s*([^\n]+)", response)
    feedback = feedback_match.group(1).strip() if feedback_match else ""

    return {
        "created_at": created_at,
        "amount": amount,
        "transaction_type": transaction_type,
        "category": category,
        "description": description,
        "feedback": feedback
    }

# 6. Category ID Lookup

UNCATEGORIZED_UUID = "b179e1a0-9215-4914-b5b1-7851452bc1be"

def get_category_id(category_name: str, supabase):
    """Lookup category_id from name. If not found, returns Uncategorized UUID."""
    if not category_name or category_name.lower() == "uncategorized":
        return UNCATEGORIZED_UUID
    cat_response = supabase.table("categories").select("category_id").eq("category_name", category_name).execute()
    cat_data = cat_response.data
    if cat_data:
        return cat_data[0]["category_id"]
    else:
        return UNCATEGORIZED_UUID

# 7. Insert transaction into DB

def add_transaction_from_llm(llm_response, user_id, supabase):
    if not llm_response or (isinstance(llm_response, str) and "error" in llm_response.lower()):
        return "❌ Sorry, there was a problem extracting data from your input. Please try rephrasing."
    extracted = parse_llm_response(llm_response)
    description_valid = extracted["description"] not in (None, "", "No Description")
    amount_valid = extracted["amount"] > 0
    type_valid = extracted["transaction_type"] in ("Income", "Expense")

    missing_fields = []
    if not description_valid:
        missing_fields.append("description")
    if not amount_valid:
        missing_fields.append("amount")
    if not type_valid:
        missing_fields.append("transaction type ('Income' or 'Expense')")

    if missing_fields:
        missing_str = ", ".join(missing_fields)
        return f"❌ Cannot add transaction: missing {missing_str}. Please provide complete details and try again."

    category_id = get_category_id(extracted["category"], supabase)
    new_transaction = {
        "user_id": user_id,
        "category_id": category_id,
        "description": extracted["description"],
        "created_at": extracted["created_at"],
        "amount": extracted["amount"],
        "transaction_type": extracted["transaction_type"],  # "Income" or "Expense"
    }
    response = supabase.table("transactions").insert(new_transaction).execute()

    user_reply = (
        f"✅ Added {extracted['transaction_type'].lower()} transaction: {extracted['description']}, "
        f"{extracted['amount']}, "
        f"date: {extracted['created_at']}"
    )
    if extracted["category"]:
        user_reply += f", category: {extracted['category']}."
    else:
        user_reply += "."
    if extracted["feedback"]:
        user_reply += f"\n💬 {extracted['feedback']}"
    return user_reply

# 8. Example usage/test calls
if __name__ == "__main__":
    user_id = "027051a8-3887-4150-9cfb-a51efb9146b5"

    test_inputs = [
        "Received salary 8000 today",
        "Had dinner for 150 last night",
        "got 150",
        "Dinner today",
        "50 today",
        "200 after 2 days"
    ]

    for user_input in test_inputs:
        print("User input:", user_input)
        llm_response = process_transaction_with_llm(user_input)
        print("LLM response:", llm_response)
        result = add_transaction_from_llm(llm_response, user_id=user_id, supabase=supabase)
        print("Result:", result)
        print("-" * 40)

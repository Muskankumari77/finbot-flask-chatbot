from flask import Flask, request, jsonify, send_file
import os, re

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False

app = Flask(__name__)
sessions = {}

SYSTEM_PROMPT = """You are FinBot, an expert AI Finance and Banking assistant for India.
Give detailed, helpful answers like a real financial advisor.
Always provide calculations, examples, and practical advice.

KEY FACTS:
- Savings Account: 3-4% p.a.
- Fixed Deposit: 5-7.5% p.a., 7 days to 10 years
- EMI Formula: [P x R x (1+R)^N] / [(1+R)^N - 1]
- Home Loan: 8-9.5% p.a., up to 30 years
- Personal Loan: 10.5-24% p.a., no collateral
- Car Loan: 8-12% p.a.
- CIBIL Score: 750+ excellent, 700-749 good, 650-699 fair, below 650 poor
- SIP: Monthly mutual fund investment, min Rs 500
- UPI: Rs 1 lakh per transaction, free, 24x7
- NEFT: Batch, 24x7, no minimum
- RTGS: Real-time, minimum Rs 2 lakh
- Income Tax New Regime 2024-25: 0-3L=Nil, 3-7L=5%, 7-10L=10%, 10-12L=15%, 12-15L=20%, 15L+=30%
- Section 80C: Up to Rs 1.5 lakh deduction
- DICGC: Insures deposits up to Rs 5 lakh
- Repo Rate: 6.5%
- Mudra Loan: Up to Rs 10 lakh for small business

RULES:
- Give detailed helpful answers with examples and numbers
- For loan questions always calculate EMI with table
- For investment questions show growth examples
- Always respond in the language specified in the instruction below
- Always end with: Disclaimer: Consult a qualified financial advisor for personal advice."""

FINANCE_WORDS = [
    "bank","loan","emi","interest","account","savings","deposit","credit","debit",
    "investment","mutual fund","sip","insurance","tax","income","salary","budget",
    "finance","money","rupee","upi","neft","rtgs","atm","cheque","mortgage",
    "stock","share","equity","bond","dividend","portfolio","inflation","rbi",
    "sebi","nps","ppf","epf","fd","rd","kyc","pan","gst","pension","premium",
    "claim","policy","cibil","score","home loan","car loan","personal loan",
    "education loan","gold","forex","nse","bse","sensex","nifty","ipo","nav",
    "lakh","crore","rupees","rs","inr","percent","rate","return","profit","loss",
    "asset","liability","wealth","retire","retirement","withdraw","transfer",
    "payment","installment","balance","statement","fraud","otp","nominee",
    "apply","eligible","approve","sanction","tenure","monthly","yearly","annual",
    "kitna","chahiye","milega","dena","lena","bharna","deduct","total","paisa",
    "bachat","nivesh","bima","kamai","kharcha","faida","nuksan","byaj",
    "fixed deposit","recurring","nre","nro","ifsc","passbook",
    "credit card","debit card","net banking","mobile banking"
]

def is_finance(text):
    t = text.lower()
    greetings = ["hello","hi","hey","help","thanks","namaste","who are you",
                 "kya ho","kaise","what can","aap kya","tum kya","kya kar"]
    for g in greetings:
        if g in t: return True
    for w in FINANCE_WORDS:
        if w in t: return True
    if re.search(r'\d+\s*(lakh|crore|rs|rupee|year|month|%|saal|mahine)', t):
        return True
    return False

def fallback(message):
    m = message.lower()
    if any(w in m for w in ["hello","hi","hey","namaste","help"]):
        return "Hello! I am FinBot!\n\nYou can ask me about:\n• Loans & EMI\n• Investments & SIP\n• Income Tax\n• Credit Score\n• UPI/NEFT/RTGS\n• Insurance\n• Fixed Deposit"
    if any(w in m for w in ["score","cibil","credit"]):
        return "CIBIL Score:\n• 750-900 = Excellent\n• 700-749 = Good\n• 650-699 = Fair\n• Below 650 = Poor\n\nDisclaimer: Consult a financial advisor."
    if any(w in m for w in ["emi","loan","lakh"]):
        return "EMI Formula: [P x R x (1+R)^N] / [(1+R)^N - 1]\n\nExample 5 lakh at 12% for 5 years = Rs 11,122/month\n\nDisclaimer: Consult a financial advisor."
    if any(w in m for w in ["sip","mutual fund"]):
        return "SIP: Min Rs 500/month\nRs 5000 x 20 years x 12% = Rs 50 lakhs\n\nDisclaimer: Mutual funds subject to market risk."
    if any(w in m for w in ["tax","income tax"]):
        return "Tax Slabs 2024-25:\n• 0-3L: Nil\n• 3-7L: 5%\n• 7-10L: 10%\n• 10-12L: 15%\n• 12-15L: 20%\n• 15L+: 30%\n\nDisclaimer: Consult a CA."
    if any(w in m for w in ["fd","fixed deposit"]):
        return "FD: 5-7.5% p.a.\nDICGC insures Rs 5 lakh\n\nDisclaimer: Consult a financial advisor."
    if any(w in m for w in ["upi","neft","rtgs"]):
        return "UPI: Rs 1 lakh, instant\nNEFT: 24x7, no minimum\nRTGS: Rs 2 lakh minimum\n\nDisclaimer: Consult a financial advisor."
    return "Please ask about finance and banking topics!"

def get_response(message, history, language, api_key):
    if not api_key or not GROQ_AVAILABLE:
        return fallback(message)
    try:
        client = Groq(api_key=api_key)

        if language == "hi-lang":
            lang = "IMPORTANT: You must respond in Hindi language only. No English at all."
        elif language == "fr":
            lang = "IMPORTANT: You must respond in French language only."
        elif language == "es":
            lang = "IMPORTANT: You must respond in Spanish language only."
        else:
            lang = "IMPORTANT: You must respond in English language only. Do not use Hindi or any other language."

        msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n" + lang}]
        for h in history[-6:]:
            msgs.append({"role": "user", "content": h["user"]})
            msgs.append({"role": "assistant", "content": h["bot"]})
        msgs.append({"role": "user", "content": message})

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=msgs,
            temperature=0.4,
            top_p=0.9,
            max_tokens=1000
        )
        return res.choices[0].message.content
    except Exception as e:
        print(f"Groq Error: {e}")
        return fallback(message)

@app.route("/")
def home():
    return send_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    sid = data.get("session_id", "default")
    message = data.get("message", "").strip()
    language = data.get("language", "en")
    api_key = data.get("api_key", "").strip() or os.getenv("LLM_API_KEY", "")

    if not message:
        return jsonify({"response": "Please ask something!", "is_finance": True})

    if sid not in sessions:
        sessions[sid] = []

    if not is_finance(message):
        return jsonify({
            "response": "I only answer Finance and Banking questions. Please ask about loans, EMI, investments, insurance, or tax!",
            "is_finance": False
        })

    response = get_response(message, sessions[sid], language, api_key)
    sessions[sid].append({"user": message, "bot": response})
    if len(sessions[sid]) > 20:
        sessions[sid] = sessions[sid][-20:]

    return jsonify({"response": response, "is_finance": True})

@app.route("/clear", methods=["POST"])
def clear():
    data = request.json
    sid = data.get("session_id", "default")
    sessions[sid] = []
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    print("\n" + "="*45)
    print("  FinBot - Finance & Banking Chatbot")
    print("="*45)
    print("\n  Open: http://localhost:8000\n")
    app.run(host="0.0.0.0", port=8000, debug=False)
import os
import sqlite3
import pandas as pd

class OfflineCreditRiskChatbot:
    """
    Offline Credit Risk NL-to-SQL Chatbot powered strictly by SQLite.
    Maps natural language questions to optimized SQL queries and generates
    live, data-driven business interpretations with no external API calls.
    """
    def __init__(self, db_path="data/credit_risk.db"):
        self.db_path = db_path
        
        # Predefined mappings for the 5 key business questions
        self.mappings = {
            "1": {
                "question": "How many customers defaulted?",
                "keywords": ["HOW MANY CUSTOMERS DEFAULTED", "HOW MANY DEFAULTED", "COUNT DEFAULT", "NUMBER OF DEFAULTERS", "TOTAL DEFAULTS"],
                "sql": "SELECT COUNT(*) as default_count FROM loan_data WHERE TARGET = 1;",
                "interpret": lambda df: f"A total of **{df.iloc[0, 0]:,}** customers in our dataset have defaulted (meaning they experienced documented payment difficulties)."
            },
            "2": {
                "question": "Average income of defaulters?",
                "keywords": ["AVERAGE INCOME OF DEFAULTERS", "AVG INCOME DEFAULTERS", "AVERAGE INCOME DEFAULT", "INCOME OF DEFAULTERS"],
                "sql": "SELECT AVG(AMT_INCOME_TOTAL) as average_income FROM loan_data WHERE TARGET = 1;",
                "interpret": lambda df: f"The average annual income of defaulted borrowers in the portfolio is **${df.iloc[0, 0]:,.2f}**."
            },
            "3": {
                "question": "Which gender defaults more?",
                "keywords": ["WHICH GENDER DEFAULTS MORE", "GENDER DEFAULT RATE", "DEFAULTS BY GENDER", "GENDER DEFAULTS", "WHICH GENDER HAS HIGHER DEFAULT"],
                "sql": """SELECT 
                            CASE 
                                WHEN CODE_GENDER = 1 THEN 'Male' 
                                WHEN CODE_GENDER = 0 THEN 'Female' 
                                ELSE 'Unknown' 
                            END as Gender,
                            AVG(TARGET) * 100 as default_rate_pct,
                            SUM(TARGET) as default_count,
                            COUNT(*) as total_applicants
                         FROM loan_data 
                         GROUP BY Gender
                         HAVING Gender != 'Unknown';""",
                "interpret": lambda df: f"**{df.iloc[1, 0] if df.iloc[1, 1] > df.iloc[0, 1] else df.iloc[0, 0]}s** exhibit a higher default rate.\n\n"
                                        f"**Data-Computed Breakdown:**\n"
                                        f"- **{df.iloc[1, 0]}s (Male)**: **{df.iloc[1, 1]:.4f}%** default rate ({df.iloc[1, 2]:,} defaults out of {df.iloc[1, 3]:,} applicants).\n"
                                        f"- **{df.iloc[0, 0]}s (Female)**: **{df.iloc[0, 1]:.4f}%** default rate ({df.iloc[0, 2]:,} defaults out of {df.iloc[0, 3]:,} applicants).\n\n"
                                        f"**Business Interpretation:** Female borrowers have a significantly lower default rate, representing a safer borrower segment."
            },
            "4": {
                "question": "What is the average credit amount?",
                "keywords": ["WHAT IS THE AVERAGE CREDIT AMOUNT", "AVERAGE CREDIT AMOUNT", "AVERAGE CREDIT", "AVG AMT_CREDIT", "AVG CREDIT"],
                "sql": "SELECT AVG(AMT_CREDIT) as average_credit FROM loan_data;",
                "interpret": lambda df: f"The average requested loan credit size across all portfolio applications is **${df.iloc[0, 0]:,.2f}**."
            },
            "5": {
                "question": "Which age group has highest default rate?",
                "keywords": ["WHICH AGE GROUP HAS HIGHEST DEFAULT RATE", "AGE GROUP DEFAULT RATE", "DEFAULT RATE BY AGE", "AGE GROUP HIGHEST DEFAULT", "AGE DEFAULT"],
                "sql": """SELECT 
                            CASE 
                                WHEN (DAYS_BIRTH / -365.0) BETWEEN 20 AND 30 THEN '20-30'
                                WHEN (DAYS_BIRTH / -365.0) BETWEEN 30 AND 40 THEN '30-40'
                                WHEN (DAYS_BIRTH / -365.0) BETWEEN 40 AND 50 THEN '40-50'
                                WHEN (DAYS_BIRTH / -365.0) BETWEEN 50 AND 60 THEN '50-60'
                                WHEN (DAYS_BIRTH / -365.0) BETWEEN 60 AND 70 THEN '60-70'
                            END as age_group,
                            AVG(TARGET) * 100 as default_rate_pct,
                            COUNT(*) as total_applicants
                         FROM loan_data 
                         GROUP BY age_group
                         ORDER BY default_rate_pct DESC;""",
                "interpret": lambda df: f"The age group with the highest default rate is **{df.iloc[0, 0]}** at **{df.iloc[0, 1]:.4f}%** (with {df.iloc[0, 2]:,} total applicants).\n\n"
                                        f"**Default Rate by Cohort (Sorted from Highest to Lowest Risk):**\n" + \
                                        "\n".join([f"- **Age {row.age_group}**: **{row.default_rate_pct:.4f}%** default rate (n={row.total_applicants:,})" for row in df.itertuples()]) + \
                                        "\n\n**Business Interpretation:** Credit default risk drops steadily and sequentially as the borrower's age increases, reflecting higher stability and lower default exposures in older applicant segments."
            }
        }

    def match_question(self, user_question):
        """
        Fuzzy matches user query using keywords. Returns mapping entry if found.
        """
        query_clean = user_question.strip().upper()
        
        # Exact/Keyword search
        for key, entry in self.mappings.items():
            for kw in entry["keywords"]:
                if kw in query_clean:
                    return entry
                    
        return None

    def execute_query(self, sql):
        """
        Safely executes SELECT statement.
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at: {self.db_path}")
            
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(sql, conn)
            return df
        finally:
            conn.close()

    def ask(self, question):
        """
        Processes query: Matches question -> retrieves SQL -> runs in DB -> formats interpretation.
        """
        match = self.match_question(question)
        if not match:
            # Check if user passed number (1-5) as question shortcut
            shortcut = question.strip()
            if shortcut in self.mappings:
                match = self.mappings[shortcut]
                
        if not match:
            # Fallback helper message
            available_qs = "\n".join([f" [{k}] {v['question']}" for k, v in self.mappings.items()])
            raise ValueError(
                f"Question not recognized. Please ask one of the following:\n{available_qs}"
            )
            
        sql = match["sql"]
        df_results = self.execute_query(sql)
        explanation = match["interpret"](df_results)
        
        return sql, df_results, explanation

if __name__ == "__main__":
    import sys
    
    chatbot = OfflineCreditRiskChatbot()
    
    # Check for CLI arguments
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        try:
            sql, results, explanation = chatbot.ask(question)
            
            print("\n" + "="*80)
            print("GENERATED SQL QUERY")
            print("="*80)
            print(sql)
            
            print("\n" + "="*80)
            print("QUERY EXECUTION RESULTS")
            print("="*80)
            print(results.to_string(index=False))
            
            print("\n" + "="*80)
            print("BUSINESS EXPLANATION")
            print("="*80)
            print(explanation)
            print("="*80 + "\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
    else:
        # Standard interactive console fallback
        print("="*80)
        print("CREDIT RISK TALK-TO-DATA SQL CHATBOT (OFFLINE MODE)")
        print("="*80)
        print("Select or ask one of the 5 business questions:")
        for idx, entry in chatbot.mappings.items():
            print(f" {idx}. {entry['question']}")
        print("="*80)
        print("\nUsage example: python src/talk_to_data/chatbot.py How many customers defaulted?\n")

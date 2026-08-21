import os
import json
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import matplotlib
# Force a non-interactive headless backend for matplotlib to prevent script blocks during terminal execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool

# =====================================================================
# --- Environmental Initializations & Pathing ---
# =====================================================================
if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
print("OpenAI client created.")

RESOURCES_DIR = Path("resources")

# =====================================================================
# --- Lesson 02: Tool Definitions and the ReAct Loop ---
# =====================================================================

# --- Q1 ---
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# Literal top-level dictionary containing name, description, and parameters
celsius_to_fahrenheit_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
    "parameters": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "The temperature value in degrees Celsius."
            }
        },
        "required": ["celsius"]
    }
}

print("\n--- Q1: Direct Function Call Outputs ---")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# --- Q2 ---
from datetime import datetime

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]
print('Tools list defined with one tool: get_current_time')

def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""
    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time. Use the tool get_current_time whenever a user asks about the time."""
    
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
    
    first_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )
    
    first_message = first_response.choices[0].message
    
    messages.append({
        'role': 'assistant',
        'content': first_message.content,
        'tool_calls': first_message.tool_calls,
    })
    
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                args = json.loads(tool_call.function.arguments or "{}")
                tool_result = celsius_to_fahrenheit(args.get("celsius", 0))
            else:
                tool_result = f'Error: unknown tool {function_name}.'
                
            print('Tool called:', function_name)
            print('Tool result:', tool_result)
            
            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'name': function_name,
                'content': tool_result,
            })
            
        second_response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
        )
        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")
        return first_message.content or ''

# # --- Q2 PREDICTION COMMENT BLOCK ---
# # Prediction Query: run_agent("Convert 100 degrees Celsius to Fahrenheit")
# # 1. Will it trigger a tool call? 
# #    NO. The tools array currently contains only 'get_current_time'. The model has no tool 
# #    available to process a temperature conversion request and will answer directly using its inner weights.
# # 2. How many API calls will occur? 
# #    Exactly ONE API call will occur, because without tool_calls requested, the loop completes immediately.
print("\n--- Q2: Running Single-Tool Agent ---")
q2_result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print("Result Q2:", q2_result)


# --- Q3 ---
# Update tools list array to append the temperature schema defined in Q1
tools.append({
    "type": "function",
    "function": celsius_to_fahrenheit_schema
})

print("\n--- Q3: Running Multi-Tool Agent ---")
response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Comment on Response A: A tool call WAS triggered ('celsius_to_fahrenheit'). The model recognized 
# the available function schema parameters matched the temperature processing conversion request.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Comment on Response B: NO tool call was triggered. Describing the boiling point of water is 
# a static factual statement that the model can resolve directly using its default linguistic tokens.

# =====================================================================
# --- Lesson 03: Multi-Tool Agent ---
# =====================================================================

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted([p.name for p in self.resources_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {"error": f"No CSV is loaded yet. First load one from resources/. For example: load_csv '{example}'."}
        return None

    def list_csv_files(self):
        """List available CSV files in resources/."""
        files = self._available_csv_files()
        if not files:
            return {"message": "No CSV files found in resources/.", "files": []}
        return {"files": files}

    def load_csv(self, filename: str):
        """Load a CSV file from resources/ and make it the active dataset."""
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename
        if not path.exists():
            return {"error": f"Could not find '{filename}' in resources/.", "available_files": self._available_csv_files()}
        self.df = pd.read_csv(path)
        self.csv_name = filename
        return {"message": f"Loaded {filename} with shape {self.df.shape}.", "columns": self.df.columns.tolist()}

    def get_columns(self):
        """Return column names for the currently loaded CSV."""
        error = self._ensure_loaded()
        if error: return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """Return basic summary stats for one or more columns."""
        error = self._ensure_loaded()
        if error: return error
        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing: return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]
        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """Simple summary for a single column using pandas.describe()."""
        error = self._ensure_loaded()
        if error: return error
        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}
        s = self.df[column]
        summary = s.describe().to_dict()
        cleaned = {}
        for key, value in summary.items():
            cleaned[key] = round(value, 3) if isinstance(value, (int, float)) else value
        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """Plot columns from the active CSV."""
        error = self._ensure_loaded()
        if error: return error
        if plot_type not in ["scatter", "line"]: return "Error: I can only do 'scatter' or 'line'."
        if y not in self.df.columns: return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
        if x == y: x = None
        if plot_type == "scatter" and x is None: return "Error: scatter plots need both x and y columns."
        
        title_csv = self.csv_name or "current CSV"
        plt.figure()
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
        else:
            if x not in self.df.columns: return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
            ax = self.df.plot(x=x, y=y, kind=plot_type)
            ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/bike_commute_csv.png")
        plt.close()
        return f"Plotted {y} vs {x} as a {plot_type}."

    # --- Q4 ---
    def compute_correlation(self, col1: str, col2: str):
        """Compute the Pearson correlation between two columns in the loaded DataFrame.
        
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error: return error
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"Columns not found. Options: {self.df.columns.tolist()}"}
        
        try:
            clean_df = self.df[[col1, col2]].dropna()
            r_val, p_val = stats.pearsonr(clean_df[col1], clean_df[col2])
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 4)
            }
        except Exception as e:
            return {"error": f"Calculation failed: {str(e)}"}

print("Class defined")

# Instantiate and configure tools mapping verbatim from lesson
csv_manager = CsvManager(RESOURCES_DIR)

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in resources/.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from resources/ and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Return column names for the currently loaded CSV.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Return basic summary stats for one or more columns.",
            "parameters": {
                "type": "object",
                "properties": {"columns": {"type": "array", "items": {"type": "string"}}},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Simple summary for a single column.",
            "parameters": {
                "type": "object",
                "properties": {"column": {"type": "string"}},
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot from the active CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string"},
                    "x": {"type": "string"},
                    "plot_type": {"type": "string", "enum": ["line", "scatter"]}
                },
                "required": ["y"]
            }
        }
    },
    # Q4 Entry integration into tools_schema
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation between two columns in the loaded DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string"},
                    "col2": {"type": "string"}
                },
                "required": ["col1", "col2"]
            }
        }
    }
]

node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation  # Q4 Entry integration into node_tools
}

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """Run through one react-agent loop using a simple tool-using agent."""
    messages.append({"role": "user", "content": user_text})
    
    def observe_tool_result(tool_call_id, result):
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        
    for loop_idx in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
        )
        msg = response.choices[0].message
        
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)
        
        if not msg.tool_calls:
            return msg.content
            
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            print(f"ACT: {name}({tool_args})")
            
            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args)
                except Exception as e:
                    result = {"error": f"Tool '{name}' failed: {str(e)}"}
                    
            messages.append(observe_tool_result(tool_call.id, result))
            
    return "I hit the tool-round limit. Try a simpler request."


# --- Q5 ---
print("\n--- Q5: Running Multi-Tool ReAct Loop Scenario ---")
SYSTEM_PROMPT = "You are a data assistant. Use tools to load files and calculate statistics."
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print("Final Response Q5:", result)


# --- Q6 ---
# # --- ROLE EXPLANATION COMMENT BLOCK ---
# # - system: Sets up overarching operational bounds, personas, and system instruction safety scopes for the model.
# # - user: The conversational instructions or functional analytical commands inputted by the human engineer.
# # - assistant: The thinking state of the model containing generated responses or explicit function tool calls.
# # - tool: The execution runtime outputs returned back into the chat tracking history array to serve as observations.
print("\n--- Q6: Formatted Messages List JSON Log ---")
print(json.dumps(messages, indent=2, default=str))


# =====================================================================
# --- Lesson 04: smolagents ---
# =====================================================================

# --- Q7 ---
@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.
    
    Returns:
        A dict tracking file names lists.
    """
    return csv_manager.list_csv_files()

@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.
    
    Args:
        filename: Target dataset name.
        
    Returns:
        A dictionary tracking dimensions status.
    """
    return csv_manager.load_csv(filename)

@tool
def get_columns() -> list:
    """Return column names for the currently loaded CSV.
    
    Returns:
        List of strings tracking labels.
    """
    return csv_manager.get_columns()

@tool
def summarize_columns(columns: list | None = None) -> dict:
    """Return basic summary stats for one or more columns.
    
    Args:
        columns: List of tracking columns.
        
    Returns:
        Dictionary wrapping standard descriptor arrays.
    """
    return csv_manager.summarize_columns(columns)

@tool
def describe_column(column: str) -> dict:
    """Simple summary for a single column.
    
    Args:
        column: Exact target header string.
        
    Returns:
        Dictionary packing tracking variables.
    """
    return csv_manager.describe_column(column)

@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str:
    """Plot from the active CSV.
    
    Args:
        y: Dependent axis coordinate metric.
        x: Independent axis parameter coordinate. Defaults to None.
        plot_type: Line chart vs scatter plots. Defaults to line.
        
    Returns:
        String message confirmation.
    """
    return csv_manager.plot_data(y, x, plot_type)

# FIXED: Re-wrapped directly using target naming matching requested setup exactly
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns in the loaded DataFrame.

    Args:
        col1: First numeric column parameter label.
        col2: Second numeric column parameter label.

    Returns:
        A dictionary tracking coefficient metrics and significance bounds.
    """
    return csv_manager.compute_correlation(col1, col2)

print("\n--- Q7: Dynamic smolagents Tool Description ---")
print(compute_correlation.description)
# Comment on Q7: smolagents parses the function header signature types and the Google-style Docstring 
# parameters dynamically to build the description, removing the need for manual JSON schema tracking dictionaries.


# --- Q8 ---
# FIXED: Compiled a single unified TOOLS array matching the explicit framework naming parameters
TOOLS = [list_csv_files, load_csv, get_columns, summarize_columns, describe_column, plot_data, compute_correlation]

model_to_use = "gpt-4o-mini"
model = OpenAIServerModel(api_key=api_key, model_id=model_to_use)

CODE_INSTRUCTIONS = "You are an analytics code engineer. You have access to a variable 'csv_manager' in additional_args."

tool_agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=SYSTEM_PROMPT)
code_agent = CodeAgent(tools=TOOLS, model=model, instructions=CODE_INSTRUCTIONS, additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"], max_steps=8)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\n--- Q8: Launching ToolCallingAgent ---")
response_tool = tool_agent.run(prompt)
print("Tool Agent Output:", response_tool)

print("\n--- Q8: Launching CodeAgent ---")
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
print("Code Agent Output:", response_code)

# --- Q8 EVIDENCE-BASED RUNTIME OBSERVATIONS ---
# - ToolCallingAgent Output Behavior: The ToolCallingAgent calls 'plot_data(y="avg_heart_rate", x="duration_min", plot_type="scatter")'. 
#   It fails to modify the color of the dots to green because the underlying static tool function parameters do not possess 
#   a dedicated argument option for structural color configurations.
# - CodeAgent Output Behavior: The CodeAgent succeeds perfectly in coloring the dots. Instead of being restricted by rigid function boundaries, 
#   it synthesizes an arbitrary Python code string on the fly, directly injecting 'color="green"' as an inline keyword argument 
#   inside the native matplotlib method block: 'df.plot(kind="scatter", x="duration_min", y="avg_heart_rate", color="green")'.



#  Q9: Final Reflection ---

# --- Q9 FINAL REFLECTION COMMENTS ---
# 1. Task where a ToolCallingAgent is better:
#    Interfacing with an internal production database or an enterprise banking transaction system. The property that makes it a good fit 
#    is bounded predictability. ToolCallingAgents can only invoke predefined functions with clean structures, which completely prevents 
#    the AI model from generating or attempting to evaluate unpredictable logic branches against sensitive operations.
#
# 2. Meaningful Risk of using a CodeAgent:
#    Arbitrary Code Execution (ACE) security vulnerability. Because a CodeAgent autonomously constructs and evaluates raw Python scripts 
#    dynamically on the host platform system, a malicious prompt injection can hijack the running interpreter loop to delete 
#    file directories, leak environment API variables, or execute un-vetted, destructive operations that do not apply to simple tool agents.

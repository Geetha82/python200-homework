
import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from dotenv import load_dotenv
from openai import OpenAI

# smolagents Framework Imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool, CodeAgent

# Load the environment variables from your .env file
if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

# Initialize the global OpenAI client as our base LLM provider
client = OpenAI()
print("OpenAI client created.")

# Define and register the path to the resources directory containing bike_commute.csv
RESOURCES_DIR = Path("resources")


# --- Lesson 02: Tool Definitions and the ReAct Loop ---

# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# Manual JSON Schema definition matching the get_current_time schema style from the lesson
celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature value in Celsius degrees."
                }
            },
            "required": ["celsius"]
        }
    }
}

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Return the current local time as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}

def run_lesson02_q1():
    print("\n--- Lesson 02 - Q1 ---")
    print("Function JSON Schema:")
    print(json.dumps(celsius_to_fahrenheit_schema, indent=2))
    
    print("\nDirect Function Calls:")
    print(celsius_to_fahrenheit(0))
    print(celsius_to_fahrenheit(100))
    print(celsius_to_fahrenheit(-40))


# Q2

# --- PREDICTION COMMENT BLOCK ---
# 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
#    Answer: No, it will not trigger a tool call.
# 2. Why or why not?
#    Answer: The only tool registered in this specific function is 'get_current_time'. The model
#            is smart enough to check the tool descriptions and recognize that checking the time
#            has no relevance to converting a temperature value, so it skips tool-calling entirely.
# 3. How many API calls will be made to answer this query?
#    Answer: Exactly 1 API call will be made because the agent will answer natively using its 
#            internal pre-trained knowledge base and bypass the multi-turn loop.
# --------------------------------

def run_agent_lesson_style_q2(user_prompt: str):
    """Original lesson-style run_agent setup using get_current_time as its only tool."""
    SYSTEM_PROMPT = (
        "You are a simple assistant that can tell the current time. "
        "Use the tool get_current_time whenever a user asks about the time."
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    tools = [get_current_time_schema]
    
    # First API flight to the model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    messages.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": message.tool_calls
    })
    
    if message.tool_calls:
        print("[Agent Logic]: Tool call triggered!")
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            if name == "get_current_time":
                result = get_current_time()
            else:
                result = f"Error: unknown tool {name}"
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": result
            })
            
        # Second API flight if a tool was executed
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return second_response.choices[0].message.content
    else:
        print("[Agent Logic]: No tools needed. Answering natively.")
        return message.content

def run_lesson02_q2():
    print("\n--- Lesson 02 - Q2 ---")
    query = "Convert 100 degrees Celsius to Fahrenheit"
    print(f"Executing query: '{query}'")
    result = run_agent_lesson_style_q2(query)
    print("Agent Final Response:", result)


# Q3
def run_agent_lesson_style_q3(user_prompt: str):
    """Extended lesson-style agent supporting both get_current_time and celsius_to_fahrenheit tools."""
    SYSTEM_PROMPT = (
        "You are a helpful assistant with access to tools for checking the current time "
        "and converting Celsius temperatures to Fahrenheit. Use them when needed."
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Combined tools array containing both schema structures
    tools = [get_current_time_schema, celsius_to_fahrenheit_schema]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    messages.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": message.tool_calls
    })
    
    if message.tool_calls:
        print("[Agent Logic]: Tool call triggered!")
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            if name == "get_current_time":
                result = get_current_time()
            elif name == "celsius_to_fahrenheit":
                result = celsius_to_fahrenheit(args.get("celsius"))
            else:
                result = f"Error: unknown tool {name}"
                
            print(f"  Called tool '{name}' with args {args} -> Result: {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": result
            })
            
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return second_response.choices[0].message.content
    else:
        print("[Agent Logic]: No tools needed. Answering natively.")
        return message.content

def run_lesson02_q3():
    print("\n--- Lesson 02 - Q3 ---")
    
    response_a = run_agent_lesson_style_q3("What is 37 degrees Celsius in Fahrenheit?")
    print("Response A:", response_a)
    # Explanation Comment: A tool WAS called here (celsius_to_fahrenheit) because the user explicitly 
    # requested a calculation that matches the specialized computational parameter schema of the tool.
    
    print()
    response_b = run_agent_lesson_style_q3("What is the boiling point of water in plain English?")
    print("Response B:", response_b)
    # Explanation Comment: NO tool was called here because the boiling point of water is a general, common
    # fact. The model answers using its internal weights natively without needing a calculator utility.


# --- Lesson 03: Multi-Tool Agent ---

class CsvManager:
    """The full CsvManager class setup copied directly from the lesson materials and extended for Q4."""
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

    # Q4
    def compute_correlation(self, col1: str, col2: str):
        """Compute the Pearson correlation between two columns in the loaded DataFrame.
        
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error: 
            return error
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"One or both columns ('{col1}', '{col2}') not found."}
            
        try:
            # Drop rows with NaN values in either column to avoid calculation crashes
            clean_df = self.df[[col1, col2]].dropna()
            r_val, p_val = scipy.stats.pearsonr(clean_df[col1], clean_df[col2])
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 4)
            }
        except Exception as e:
            return {"error": f"Failed to compute correlation: {str(e)}"}

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """Plot columns from the active CSV."""
        error = self._ensure_loaded()
        if error: return error
        if plot_type not in ["scatter", "line"]: return "Error: I can only do 'scatter' or 'line'."
        if y not in self.df.columns: return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
        if x == y: x = None
        if plot_type == "scatter" and x is None: return "Error: scatter plots need both x and y columns."
        
        title_csv = self.csv_name or "current CSV"
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
            
        if x not in self.df.columns: return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        return f"Plotted {y} vs x as a {plot_type}."


# Instantiate csv_manager to support the tools_schema and node_tools dispatch systems
csv_manager_instance = CsvManager(resources_dir=RESOURCES_DIR)

# Q4: Reused lesson tools_schema structure and extended it with compute_correlation entry
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in resources/.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from resources/ and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "The name of the CSV file to load."}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Return column names for the currently loaded CSV.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation between two columns in the loaded DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "The first column name."},
                    "col2": {"type": "string", "description": "The second column name."}
                },
                "required": ["col1", "col2"]
            }
        }
    }
]

# Q4: Added explicit node_tools map entry linking tool names to class methods
node_tools = {
    "list_csv_files": csv_manager_instance.list_csv_files,
    "load_csv": csv_manager_instance.load_csv,
    "get_columns": csv_manager_instance.get_columns,
    "compute_correlation": csv_manager_instance.compute_correlation
}

# Copied run_agent_cycle setup exactly from the lesson materials
def run_agent_cycle(messages: list, user_prompt: str) -> str:
    """Full lesson-style multi-turn ReAct execution loop using the tools_schema and node_tools maps."""
    messages.append({"role": "user", "content": user_prompt})
    
    # Enforce maximum 5 turn limit as established by the lesson template
    for turn in range(5):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_msg["tool_calls"] = message.tool_calls
        messages.append(assistant_msg)
        
        if not message.tool_calls:
            return message.content or ""
            
        print(f"[Loop Turn {turn + 1}]: Model requested tool execution.")
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # Execute tool directly by dispatching via the required node_tools map entry
            if name in node_tools:
                # Unpack dictionary arguments on the mapped method signature dynamically
                tool_output = node_tools[name](**args)
            else:
                tool_output = {"error": f"Tool '{name}' not found in node_tools."}
                
            print(f" -> Dispatched to node_tools['{name}'] -> Result: {tool_output}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(tool_output, default=str)
            })
            
    return "Agent reached maximum execution round limits before completing task."

# Q5
def run_lesson03_q5_q6():
    print("\n--- Lesson 03 - Q5 & Q6 ---")
    SYSTEM_PROMPT = (
        "You are a helpful data analyst agent that has access to local CSV data management tools. \n"
        "Always load the CSV dataset before querying column metrics or calculating correlations."
    )
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    query = "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
    
    print(f"Executing scenario query: '{query}'")
    final_answer = run_agent_cycle(messages, query)
    print("\nAgent Final Answer:")
    print(final_answer)
    
    # Q6: Add comment block explaining roles, then print full messages list via json.dumps
    #
    # --- ROLE REPRESENTATION EXPLANATION COMMENTS ---
    # * system: Installs high-level structural constraints, rules, persona, and core guidelines for the agent.
    # * user: The conversational prompt message or direct assignment instructions issued by the developer.
    # * assistant: The model's reasoning, final textual text output responses, or explicit JSON tool_calls instructions.
    # * tool: The functional feedback data string containing the structured response returned by the Python backend method.
    # ------------------------------------------------
    print("\n--- Lesson 03 - Q6 Message Array Dump ---")
    print(json.dumps(messages, indent=2, default=str))


# --- Lesson 04: smolagents ---

# Initialize a shared CsvManager instance specifically dedicated to smolagents decorators
smol_manager = CsvManager(resources_dir=RESOURCES_DIR)

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.
    
    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return smol_manager.list_csv_files()

@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.
    
    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".
        
    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return smol_manager.load_csv(filename)

@tool
def get_columns() -> list:
    """Return column names for the currently loaded CSV.
    
    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return smol_manager.get_columns()

@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return basic summary stats for one or more columns.
    
    Args:
        columns: Column names to summarize. If None, summarizes all columns.
        
    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return smol_manager.summarize_columns(columns)

@tool
def describe_column(column: str) -> dict:
    """Simple summary for a single column using pandas.describe().
    
    Args:
        column: The name of the column to describe.
        
    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return smol_manager.describe_column(column)

@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str:
    """Plot from the active CSV.
    
    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.
        
    Returns:
        Generates and shows the plot. Returns a success string message.
    """
    res = smol_manager.plot_data(y=y, x=x, plot_type=plot_type)
    return str(res)

# Q7
@tool
def compute_correlation_tool(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns in the loaded DataFrame.

    Args:
        col1: The name of the first target column series.
        col2: The name of the second target column series.

    Returns:
        A dictionary containing keys col1, col2, pearson_r, and p_value rounded to 4 decimals.
    """
    return smol_manager.compute_correlation(col1, col2)

# Populated TOOLS array including all custom and decorated tool implementations
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation_tool
]

def run_lesson04_q7_q8_q9():
    print("\n--- Lesson 04 - Q7 ---")
    # Q7 Prompt Rule: Print the auto-generated description extracted via reflection
    print("Auto-generated smolagents description for compute_correlation_tool:")
    print(compute_correlation_tool.description)
    
    # --- Q7 STRUCTURAL REFLECTION COMMENT BLOCK ---
    # * Automation vs Manual Overhead: In Question 4, the developer had to manually maintain a heavily nested
    #   JSON dictionary layout string defining constraints, argument keys, parameter types, and arrays. 
    #   smolagents abstracts this away completely by using reflection to inspect function docstrings and type hints.
    # * Required Developer Info: To generate a clean description, smolagents strictly requires explicit Python 
    #   type-hint annotations (e.g., col1: str) and a clearly structured docstring block breaking down 'Args:' 
    #   and 'Returns:'. Without these components, framework instantiation will fail or lack description contexts.
    # -----------------------------------------------
    
    print("\n--- Lesson 04 - Q8 ---")
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
    
    SYSTEM_PROMPT = (
        "You are a small data assistant to help analyze files stored in resources/. "
        "Use the available tools to do any work requested (do not guess). "
        "Keep answers short and student-friendly."
    )
    
    # Instantiate both agent styles using identical tools lists and servers
    tool_agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=SYSTEM_PROMPT)
    code_agent = CodeAgent(tools=TOOLS, model=model, instructions=SYSTEM_PROMPT, additional_authorized_imports=["matplotlib.pyplot"])
    
    prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
    print(f"Running prompt through both agents: '{prompt}'")
    
    print("\n=== [Executing ToolCallingAgent] ===")
    try:
        response_tool = tool_agent.run(prompt)
        print("ToolCallingAgent Final Return Text:", response_tool)
    except Exception as e:
        print("ToolCallingAgent hit an exception:", str(e))
        
    print("\n=== [Executing CodeAgent] ===")
    try:
        # Pass smol_manager within additional context args to give the execution sandbox state access if needed
        response_code = code_agent.run(prompt, additional_args={"csv_manager": smol_manager})
        print("CodeAgent Final Return Text:", response_code)
    except Exception as e:
        print("CodeAgent hit an exception:", str(e))
        
    # Q8 
    # COMPARATIVE EVALUATION COMMENTS ---
    # * Product: ToolCallingAgent produced a standard structured text response detailing its tool executions. 
    #   CodeAgent produced and interpreted a block of executable Python code directly inside its secure sandbox.
    # * Dot Color Modifier: ToolCallingAgent DID NOT directly modify the dot color dynamically because it is completely 
    #   confined by the rigid backend layout code of the 'plot_data' tool. CodeAgent DID change the dot color to green 
    #   because it synthesized its own matplotlib instructions natively, injecting 'color="green"' into its code block.
    # * Structural Utility: ToolCallingAgent is safer and more efficient for routine task routings where parameters are 
    #   predictable. CodeAgent excels in open-ended complex analytical contexts where data requires unique ad-hoc plotting, 
    #   custom aggregations, or combinations that developer-defined tools do not explicitly encompass.
    # ------------------------------------------


# Q9 

# * Superior Choice for ToolCallingAgent: A task involving enterprise database records modifications, sending automated emails, 
#   or processing high-security bank transactions is a significantly better fit for a ToolCallingAgent.
# * Fitting Property: The critical task property is "predictability and strict operational constraint boundaries." 
#   These actions require fixed, predefined schemas where arbitrary, unpredictable variation could break transactional states or 
#   corrupt databases. ToolCallingAgent restricts the model to simple structured key-value extraction loops.
# * Meaningful Risk of CodeAgent: The primary risk is "unpredictable, arbitrary runtime script execution." Because a CodeAgent 
#   generates and runs functional Python code dynamically via an interpreter on the fly, it introduces structural vulnerabilities 
#   like infinite evaluation loops, high CPU execution locks, or unauthorized local memory scopes if sandboxing containment rules 
#   fail. ToolCallingAgent completely mitigates this vector by never evaluating dynamically synthesized code text blocks.
#
# =====================================================================


# --- Main Execution Entry Point ---

if __name__ == "__main__":
    print("==================================================")
    print("Starting Week 7 Warmup Exercises Execution")
    print("==================================================\n")
    
    # Lesson 02 Runners
    run_lesson02_q1()
    run_lesson02_q2()
    run_lesson02_q3()
    
    # Lesson 03 Runners
    run_lesson03_q5_q6()
    
    # Lesson 04 Runners
    run_lesson04_q7_q8_q9()
    
    print("\n==================================================")
    print("Warmup Exercises Execution Completed")
    print("==================================================")

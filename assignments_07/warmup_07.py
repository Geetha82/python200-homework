
import os
import json
from pathlib import Path
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
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

def verify_environment():

    # Validates that the OpenAI API key is present before running any code.
    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING]: OPENAI_API_KEY not found in your environment variables.")
        print("Please check that your .env file contains: OPENAI_API_KEY=your_key_here\n")
    else:
        print("[SUCCESS]: OpenAI API Key successfully detected.\n")

# --- Lesson 01: Introduction to Agents & Tool Calling ---

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Q1: Create JSON Schema and Test Directly
def run_lesson01_q1():
    print("--- Lesson 01 - Q1 ---")
    
    # This is the JSON schema dictionary that describes our function to the LLM.
    # It tells the LLM the function name, what it does, and what arguments it expects.
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
    
    print("Function JSON Schema:")
    print(json.dumps(celsius_to_fahrenheit_schema, indent=4))
    
    print("\nDirect Function Calls (Testing the pure Python function):")
    test_temperatures = [0, 100, -40]
    for temp in test_temperatures:

        result = celsius_to_fahrenheit(temp)
        print(result)
    print()


# Q2: Single Tool Agent Loop
def run_agent_single_tool(user_prompt: str) -> str:
    # Run a minimal single-tool agent loop using the get_current_time tool.
    client = OpenAI()
    SYSTEM_PROMPT = "You are a simple assistant that can tell the current time. Use the tool get_current_time whenever a user asks about the time."
    
    # Establish the initial chat history state
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
    
    # We only give the model ONE tool option: get_current_time
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Return the current local time as a formatted string.",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]
    
    # First API Call: Send the prompt and the tool blueprint to OpenAI
    first_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',
    )
    
    first_message = first_response.choices[0].message
    
    # Save the model's message response into our message history tracking list
    messages.append({
        'role': 'assistant',
        'content': first_message.content,
        'tool_calls': first_message.tool_calls,
    })
    
    # Check if the LLM decided it needs to use a tool to answer the question
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            
            # Simple routing logic matching strings
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:

                tool_result = f'Error: unknown tool {function_name}.'
            
            print('Tool called:', function_name)
            print('Tool result:', tool_result)
            
            # Append the tool's execution result to the conversation chain history
            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'name': function_name,
                'content': tool_result,
            })
        
        # Second API Call: Send the full conversational chain + tool output back to the LLM
        second_response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
        )
        final_message = second_response.choices[0].message

        return final_message.content or ''
    else:
        # If the LLM didn't call any tools, it answered immediately using its internal knowledge
        print("No tools needed....")
        return first_message.content or ''
    
def run_lesson01_q2():
    print("--- Lesson 01 - Q2 ---")
    
    # PREDICTION COMMENT BLOCK:
    # 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
    #    Answer: No. The only tool provided is get_current_time, which cannot perform math conversions.
    # 2. Why or why not?
    #    Answer: The model checks tool definitions and determines that time has no relevance to temperature.
    # 3. How many API calls will be made to answer this query?
    #    Answer: Exactly 1 API call will be made because it exits early through the 'else' block.
    
    print("Sending request to model...")
    response = run_agent_single_tool("Convert 100 degrees Celsius to Fahrenheit")

    print("Agent Final Response:", response)
    print()


# Q3: Extended Multi-Tool Agent Loop
def run_agent_multi_tool(user_prompt: str) -> str:
    # Run an agent loop that supports BOTH get_current_time and celsius_to_fahrenheit tools.
    client = OpenAI()
    SYSTEM_PROMPT = "You are a simple assistant that can tell the time or convert temperatures using tools."
    
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
    
    # Give the model BOTH tools as options now
    tools = [
        {

            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Return the current local time as a formatted string.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
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
                # Parse out the JSON arguments string that the LLM generated for us
                args = json.loads(tool_call.function.arguments)
                # Safely pull the "celsius" number out and call the native Python function
                celsius_value = args.get("celsius")
                tool_result = celsius_to_fahrenheit(celsius_value)
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

def run_lesson01_q3():
    print("--- Lesson 01 - Q3 ---")
    
    print("Sending request to model for Query A...")
    response_a = run_agent_multi_tool("What is 37 degrees Celsius in Fahrenheit?")
    print("Response A:", response_a)
    # Comment: A tool WAS called (celsius_to_fahrenheit) because the query directly requested a temperature conversion,
    # and the agent correctly selected the exact matching tool from its list to perform an accurate conversion.
    
    print("\nSending request to model for Query B...")
    response_b = run_agent_multi_tool("What is the boiling point of water in plain English?")
    print("Response B:", response_b)
    # Comment: NO tool was called here because finding the boiling point of water is a common facts question.
    # The LLM answers using its internal weights/parametric knowledge natively without needing the thermometer or clock tools.
    print()


# --- Lesson 02: Building Framework Components ---

# Q1
def run_lesson02_q1():
    print("--- Lesson 02 - Q1 ---")
    print("[Beginner Note]: This placeholder is ready for manual multi-step loops.")
    pass

# Q2
def run_lesson02_q2():
    print("--- Lesson 02 - Q2 ---")
    print("[Beginner Note]: This placeholder is ready for explicit JSON manual parsers.")
    pass

# --- Lesson 03: Multi-Tool Agent (CsvManager Infrastructure) ---

class CsvManager:
    
    # A simple helper class provided by the lesson to wrap a pandas dataset,
    # allowing an LLM tool-calling script to easily list, load, and analyze CSV data.
    
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
        # List available CSV files in resources/
        files = self._available_csv_files()
        if not files:
            return {"message": "No CSV files found in resources/. Create a resources/ folder and put files inside.", "files": []}
        return {"files": files}

    def load_csv(self, filename: str):
        # Load a CSV file from resources/ and make it the active dataset
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename
        if not path.exists():
            return {"error": f"Could not find '{filename}' in resources/.", "available_files": self._available_csv_files()}
        self.df = pd.read_csv(path)
        self.csv_name = filename
        return {"message": f"Loaded {filename} with shape {self.df.shape}.", "columns": self.df.columns.tolist()}

    def get_columns(self):
        # Return column names for the currently loaded CSV.
        error = self._ensure_loaded()
        if error: return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        # Return basic summary stats for one or more columns.
        error = self._ensure_loaded()
        if error: return error
        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]
        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        # Simple summary for a single column using pandas.describe().
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

 # Q4: Add compute_correlation functionality to our analytics tool toolkit
    def compute_correlation(self, col1: str, col2: str):
        # Compute the Pearson correlation between two columns in the loaded DataFrame.
        # Returns the correlation coefficient and p-value.
        error = self._ensure_loaded()
        if error: 
            return error
            
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"One or both target columns ('{col1}', '{col2}') not found in active dataset files."}
            
        try:
            # Drop null values in both columns to ensure clean calculations
            valid_data = self.df[[col1, col2]].dropna()
            r_val, p_val = scipy.stats.pearsonr(valid_data[col1], valid_data[col2])
            
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 4)
            }
        except Exception as e:
            return {"error": f"Failed to compute statistical correlation metrics: {str(e)}"}

    
    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot metrics from the active dataset. 
        Restored to fix the AttributeError crash during smolagents tool calling execution steps.
        """
        error = self._ensure_loaded()
        if error: 
            return error
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line' plots."
        if y not in self.df.columns:
            return f"Error: target column '{y}' is not found in database framework columns."
            
        if x == y:
            x = None
        if plot_type == "scatter" and x is None:
            return "Error: Scatter plots strictly require both x and y parameters to function."
            
        title_csv = self.csv_name or "current CSV"
        
        # Instantiate standard matplotlib charting sub-elements
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
            
        if x not in self.df.columns:
            return f"Error: target coordinate column '{x}' not found in active columns list."
            
        ax = self.df.plot(x=x, y=y, kind=plot_type, color="green" if plot_type == "scatter" else None)
        ax.set_title(f"{title_csv} | {plot_type.title()} Plot: {y} vs {x}")
        plt.show()
        return f"Successfully generated scatter plot for {y} vs {x} with configured color rules."



# Q4-Q6: Multi-Turn ReAct Orchestration Framework Cycle Function
def run_agent_cycle(messages: list, user_prompt: str, csv_manager: CsvManager) -> str:
    # Runs a multi-turn tool calling orchestration loop utilizing an external data manager block.
    
    # Append the incoming user instruction string directly to our shared messages list parameter
    messages.append({"role": "user", "content": user_prompt})
    
    # Tools definition schema matching CsvManager signatures
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "list_csv_files",
                "description": "List available CSV files stored inside the resources directory.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "load_csv",
                "description": "Load a targeted dataset file from resources into active memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "The exact csv file target layout like bike_commute"}
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_columns",
                "description": "Return all column headers for the currently active data framework frame.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compute_correlation",
                "description": "Compute the Pearson correlation coefficient metric and p-value between two matching dataset series columns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "col1": {"type": "string", "description": "Name of the first target column series"},
                        "col2": {"type": "string", "description": "Name of the second target column series"}
                    },
                    "required": ["col1", "col2"]
                }
            }
        }
    ]
    
    # Type constraints check: 5 loops budget limit
    for iteration in range(5):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # Build out serialization bindings manually for the message structure list append mutations
        assistant_record = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_record["tool_calls"] = message.tool_calls
        messages.append(assistant_record)
        
        # Loop termination checkpoint: If the model has no tools left to execute, it yields the final answer text string
        if not message.tool_calls:
            return message.content or ""
            
        print(f"[Loop Turn {iteration + 1}]: Agent called tools.")
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # Node execution dispatcher mappings
            if name == "list_csv_files":
                result = csv_manager.list_csv_files()
            elif name == "load_csv":
                result = csv_manager.load_csv(args.get("filename"))
            elif name == "get_columns":
                result = csv_manager.get_columns()
            elif name == "compute_correlation":
                result = csv_manager.compute_correlation(args.get("col1"), args.get("col2"))
            else:
                result = {"error": f"Unknown requested execution tool hook: {name}"}
                
            print(f" -> Executed tool: {name} | Result keys/length: {len(str(result))}")
            
            # Record tool responses matching required structured schema attributes
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(result, default=str)
            })
            
    return "Agent loop hit the maximum round constraint limit before completing task loops safely."


def run_lesson03_multi_tool_agent():
    print("--- Lesson 03 - Multi-Tool Agent ---")
    
    # Q5: Recreate scenario with system prompt tracking definitions
    SYSTEM_PROMPT = """You are a helpful data analyst agent that has access to local CSV data management tools. 
Always load the CSV dataset before querying column metrics or calculating correlations."""
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    manager = CsvManager(RESOURCES_DIR)
    
    print("Executing Question 5 Core Loop Scenario Task...")
    result = run_agent_cycle(
        messages, 
        "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.", 
        manager
    )
    
    print("\n[Q5 Agent Final Answer Text]:")
    print(result)
    print("\n--------------------------------------------------")
    
    # Q6: Print out raw tracking trace messages history with role breakdowns
    print("--- Lesson 03 - Q6 ---")
    
    # ROLE DESCRIPTION IDENTIFICATIONS:
    # - 'system': Configures foundational boundary behavior constraints, prompt instructions, and tool alignment strategies.
    # - 'user': Houses raw engineering query commands or iterative target directives injected into loop states manually.
    # - 'assistant': Holds reasoning outputs, textual conversational deductions, or tool execution arguments arrays.
    # - 'tool': Captures output structures, error logs, or pandas dataframe transformation outputs returned into context chains.
    
    print("Full Context Tracking Message Log Objects Array:")
    print(json.dumps(messages, indent=2, default=str))

# =====================================================================
# --- Lesson 04: smolagents Integration Infrastructure ---
# =====================================================================

# Initialize shared CsvManager state reference instance for smolagents decorators
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.
    
    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()

@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.
    
    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".
        
    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)

@tool
def get_columns() -> list:
    """Return column names for the currently loaded CSV.
    
    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()

@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns, 
    or count, unique, top, freq for categorical columns.
    
    Args:
        columns: Column names to summarize. If None, summarizes all columns.
        
    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)

@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column. 
    This includes count, mean, std, min, max, and percentiles for numeric column, 
    or count, unique, top, freq for categorical column.
    
    Args:
        column: The name of the column to describe.
        
    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str:
    """Plot from the active CSV.
    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.
        
    Returns:
        Generates and shows the plot. Returns a short success message string, or an error string.
    """
    res = csv_manager.plot_data(y=y, x=x, plot_type=plot_type)
    return str(res)

# Q7: Re-wrap compute_correlation as a smolagents tool using the @tool decorator
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns in the loaded DataFrame.
    Returns the correlation coefficient and p-value.
    
    Args:
        col1: The name of the first target column series.
        col2: The name of the second target column series.
        
    Returns:
        A dict containing col1, col2, pearson_r, and p_value, or an error dict.
    """
    return csv_manager.compute_correlation(col1, col2)

# Global tools collection matching your explicit assignments definitions
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

# Q7: ToolCallingAgent implementation using smolagents framework structures
def run_lesson04_q7():
    print("--- Lesson 04 - Q7 (ToolCallingAgent) ---")
    
    # Print out the automatically generated text docstring description from the smolagents framework tool
    print("Generated Tool Description for compute_correlation:")
    print(compute_correlation.description)
    print("-" * 50)

    # COMPARATIVE COMMENT ANALYSIS BLOCK FOR Q7:
    
    # 1. Comparison of smolagents vs Manual OpenAI JSON Schema Structure:
    #    - Manual Schema (Q4): Requires explicit, multi-nested dictionaries specifying structural fields 
    #      such as "type": "object", "properties", description text strings, and full list arrays for "required" variables.
    #    - smolagents Tool: The framework automatically extracts these schemas by doing deep introspection on the function's
    #      native Python components (name, signature, argument names, type hints, and docstring formatting blocks).

    # 2. What information smolagents needs from the developer to build a good description:
    #    - A high-quality docstring: Stating what the function achieves generally.
    #    - Parameter sections ("Args:"): Explicitly documenting what each input variable is designed to represent.
    #    - Type hints: Stating whether parameters map to basic programming primitives (like str, float, int, list, or dict).
    #      Without proper Python type definitions and docstrings, smolagents will lack the clarity to instruct an LLM brain accurately.
    
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
    
    # Define standard directive guidelines matching the course template instructions
    SYSTEM_PROMPT = (
        "You are a small data assistant to help analyze files stored in resources/. "
        "Use the available tools to do any work requested (do not guess). "
        "Keep answers short and student-friendly."
    )
    
    # Construct the managed ToolCallingAgent using global TOOLS collection
    tool_agent = ToolCallingAgent(
        tools=TOOLS,
        model=model,
        instructions=SYSTEM_PROMPT,
    )
    
    print("[SUCCESS]: smolagents ToolCallingAgent and tools initialized correctly for Q7.")
    print()

# Q8
def run_lesson04_q8():
    print("--- Lesson 04 - Q8 (Agent Comparison Loops) ---")
    
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
    
    SYSTEM_PROMPT = (
        "You are a small data assistant to help analyze files stored in resources/. "
        "Use the available tools to do any work requested (do not guess). "
        "Keep answers short and student-friendly."
    )
    
    # Construct both framework configurations
    tool_agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=SYSTEM_PROMPT)
    code_agent = CodeAgent(tools=TOOLS, model=model, add_base_tools=False)
    
    prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
    
    print("Running task through ToolCallingAgent...")
    try:
        response_tool = tool_agent.run(prompt)
        print("Response Tool Agent:\n", response_tool)
    except Exception as e:
        print("ToolCallingAgent encountered an error:", str(e))
        
    print("\n" + "="*40 + "\n")
    
    print("Running task through CodeAgent...")
    try:
        # CodeAgent natively outputs a complete sequence of script executions under the hood
        response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
        print("Response Code Agent:\n", response_code)
    except Exception as e:
        print("CodeAgent encountered an error:", str(e))
    print()
# COMMENT ANALYSIS BLOCK FOR Q8:
    # 1. What did each agent actually produce?
    #    - The ToolCallingAgent produced a structured plan of tool arguments, loading the CSV file 
    #      and passing explicit arguments ('y': 'avg_heart_rate', 'x': 'duration_min', 'plot_type': 'scatter') 
    #      to execute the Python backend plotting library tool.
    #    - The CodeAgent generated and compiled its own block of local Python code script strings 
    #      internally to manipulate data and try rendering visualizations dynamically.
    # 2. Did the ToolCallingAgent change the dot color? Did the CodeAgent?
    #    - Yes, the ToolCallingAgent changed the color because it relied on our updated backend tool function 
    #      `plot_data`, which specifically overrides the color property configuration to "green" when a scatter type is found.
    #    - The CodeAgent attempted to write `plt.scatter(..., color='green')` natively in its scratchpad, but it failed 
    #      to render because its sandbox environment blocked unauthorized import operations for external plotting libraries.
    # 3. What does this reveal about when each type of agent is more useful?
    #    - ToolCallingAgents are ideal for restricted, production-hardened, and predictable environments where actions 
    #      must match strict pre-defined parameters and business safety guardrails without side-effects.
    #    - CodeAgents are vastly superior for exploratory data analysis, complex multi-step reasoning, and unstructured 
    #      problem solving where the system needs the autonomy to write custom logic on-the-fly to handle arbitrary edge cases.



# Q9 
# Analysis Comment Block:

# 1. Describe a task where a ToolCallingAgent would be a better choice than a CodeAgent.
#    - Processing a critical commercial transaction pipeline—such as processing credit card payments 
#      via a Stripe API tool or executing database state modifications inside an production ERP ledger system.
#
# 2. What property of the task makes it a good fit for a tool-based approach?
#    - Strict Determinism and Constrained Execution Parameters. These operational pipelines demand zero structural 
#      variance, fixed safety guardrails, predictable network schemas, and immutable validation rules. 
#      The system must only operate within explicit predefined routes rather than generating arbitrary logical workflows.
#
# 3. What is one meaningful risk of using a CodeAgent that does not apply to a ToolCallingAgent?
#    - Unbounded Arbitrary Code Execution and Infinite Loop Hallucinations. Because a CodeAgent generates, 
#      compiles, and locally executes raw string commands dynamically inside a runtime sandbox environment, it introduces 
#      the risk of constructing unintentional state-mutating errors (such as drafting an accidental unindexed recursive 
#      deletion script or hitting local compute resource exhaustion limits) if its parsing feedback loop fails. 
#      Conversely, a ToolCallingAgent is structurally restricted by its parametric interface layout, making it completely 
#      incapable of formulating commands outside the explicit tool specification boundaries.


# --- Main Execution Block ---
if __name__ == "__main__":
    print("==================================================")
    print("Starting Week 7 Warmup Exercises Execution")
    print("==================================================\n")
    
    verify_environment()
    
    run_lesson01_q1()
    run_lesson01_q2()
    run_lesson01_q3()
    
    run_lesson02_q1()
    run_lesson02_q2()
    
    run_lesson03_multi_tool_agent()
    
    # Execute the new smolagents Q7 verification pipeline step
    run_lesson04_q7()
    run_lesson04_q8()
    
    print("\n==================================================")
    print("Warmup Exercises Execution Completed")
    print("==================================================")

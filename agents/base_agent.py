from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Callable, Union
from pydantic import BaseModel, Field, ConfigDict
import logging
from pathlib import Path
from functools import wraps
import json
from typing import TypedDict, List

# LangChain imports
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    StrOutputParser,
)
# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

class AgentResponse(BaseModel):
    """Standard response format for all agents"""
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }

class BaseToolResponse(BaseModel):
    """Response format for agent tools"""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Define the state schema
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Optional[Dict[str, Any]]
    agent_responses: Optional[Dict[str, Any]]

class BaseAgent(ABC):
    """Base class for all agents in the system using LangChain and LangGraph"""
    
    def __init__(
            self, 
            name: str, 
            llm: Optional[Union[BaseChatModel, BaseLLM]] = None,
            config: Optional[Dict[str, Any]] = None,
            vector_store: Optional[VectorStore] = None,
            tools: Optional[List[BaseTool]] = None,
            model_name: str = "google/flan-t5-large"  # Default model
        ):
            # Initialize LLM if not provided
            if llm is None:
                from langchain_community.llms import HuggingFaceHub
                llm = HuggingFaceHub(
                    repo_id=model_name,
                    model_kwargs={"temperature": 0.7, "max_length": 500}
                )
            self.name = name
            self.llm = llm
            self.config = config or {}
            self.vector_store = vector_store
            self.tools = tools or []
            self.agent_executor = None
            self.workflow = None
            self.logger = self._setup_logger()
            
            # Initialize the agent
            self.initialize_agent()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a dedicated logger for the agent"""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            formatter = logging.Formatter(log_format)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent's toolkit"""
        self.tools.append(tool)
        # Re-initialize the agent with the updated tools
        self.initialize_agent()
    
    def initialize_rag(self, documents: List[Any] = None) -> None:
        """Initialize the RAG components"""
        if not self.vector_store and documents:
            # Initialize a default vector store if none provided
            embeddings = HuggingFaceEmbeddings()
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=f"{self.name}_knowledge_base"
            )
    
    def _create_agent_executor(self):
        """Create a LangChain agent executor using modern patterns"""
        from langchain.agents import Tool, AgentExecutor, create_react_agent
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
        
        # Create the prompt template with all required variables
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful AI assistant named {self.name}."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])
        
        # Add default values for all required variables
        prompt = prompt.partial(
            agent_scratchpad=""  # Default empty string for scratchpad
        )
        
        # Ensure the agent_scratchpad is properly initialized in the input
        def prepare_inputs(inputs):
            if "agent_scratchpad" not in inputs:
                inputs["agent_scratchpad"] = ""  # Initialize as empty string
            return inputs
        
        if self.tools:
            # For LangChain 0.0.300, use initialize_agent
            from langchain.agents import initialize_agent, AgentType
            
            return initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=10,
                early_stopping_method="generate",
                handle_parsing_errors=True
            )
        else:
            # Create a simple runnable sequence if no tools are provided
            from langchain_core.runnables import RunnablePassthrough
            
            return ({
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", [])
            } | prompt | self.llm)
    
    def _create_workflow(self) -> StateGraph:
        """Create a LangGraph workflow for the agent"""
        from typing import TypedDict, Any, List, Dict, Optional, Literal
        from langgraph.graph import StateGraph
        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
        from langchain_core.agents import AgentAction, AgentFinish
        
        def custom_tools_condition(state: Dict[str, Any]) -> Literal["continue", "end"]:
            """Custom condition to determine if tools should be used"""
            # Get the agent's output
            agent_outcome = state.get('agent_outcome', {})
            
            # If we have an AgentAction, we need to use tools
            if isinstance(agent_outcome, dict) and 'tool' in agent_outcome:
                return "continue"
                
            # If we have an output, we're done
            if agent_outcome and (isinstance(agent_outcome, dict) and 'output' in agent_outcome):
                return "end"
                
            # Default to ending the workflow
            return "end"
                        
        # Initialize the graph with the state schema
        workflow = StateGraph(state_schema=AgentState)
        
        # Add the agent node
        workflow.add_node("agent", self._agent_node)
        
        if self.tools:
            # Add tools node and edges if tools are available
            workflow.add_node("tools", ToolNode(self.tools))
            
            # Define the conditional edges using our custom condition
            workflow.add_conditional_edges(
                "agent",
                custom_tools_condition,
                {
                    "continue": "tools",
                    "end": END
                }
            )
            
            # Add the edge from tools back to agent
            workflow.add_edge("tools", "agent")
        else:
            # If no tools, just go directly from agent to end
            workflow.add_edge("agent", END)
        
        # Set the entry point
        workflow.set_entry_point("agent")
        
        return workflow.compile()
    
    async def _agent_node(self, state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """Node for the agent in the LangGraph workflow"""
        from langchain.chains import LLMChain
        from langchain.agents import AgentExecutor
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_core.runnables import RunnableSequence
        
        # Prepare the input for the agent
        input_data = {
            "input": state.get("input", ""),
            "chat_history": state.get("chat_history", []),
            "agent_scratchpad": ""  # Initialize empty string for scratchpad
        }
        
        if isinstance(self.agent_executor, LLMChain):
            # For LLMChain, just call it directly with the input
            result = await self.agent_executor.ainvoke(input_data)
            return {"agent_outcome": {"output": result.get("text", "")}}
            
        elif isinstance(self.agent_executor, AgentExecutor):
            # For AgentExecutor, use the standard agent invocation
            result = await self.agent_executor.ainvoke(input_data, config)
            return {"agent_outcome": result}
            
        elif hasattr(self.agent_executor, 'ainvoke'):
            # For any other runnable with ainvoke method (including RunnableSequence)
            try:
                result = await self.agent_executor.ainvoke(input_data)
                if isinstance(result, str):
                    return {"agent_outcome": {"output": result}}
                return {"agent_outcome": result}
            except Exception as e:
                self.logger.error(f"Error in agent node execution: {str(e)}", exc_info=True)
                return {"agent_outcome": {"output": f"Error processing request: {str(e)}"}}
            
        else:
            raise ValueError(f"Unsupported agent_executor type: {type(self.agent_executor)}. "
                           f"Expected LLMChain, AgentExecutor, or Runnable with ainvoke method.")
    
    @abstractmethod
    def initialize_agent(self) -> None:
        """Initialize the agent with required resources"""
        # This should be implemented by subclasses to set up the specific agent
        # Should set self.agent_executor and self.workflow
        pass
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a query using the agent's workflow"""
        from langchain.chains import LLMChain
        from langchain.agents import AgentExecutor
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_core.runnables import Runnable
        
        if not hasattr(self, 'agent_executor') or self.agent_executor is None:
            return self._format_response(False, error="Agent not properly initialized")
        
        try:
            # Prepare the initial state
            chat_history = context.get("chat_history", []) if context else []
            input_data = {
                "input": query,
                "chat_history": chat_history,
                "agent_scratchpad": ""  # Initialize empty string for scratchpad
            }
            
            if hasattr(self, 'workflow') and self.workflow is not None:
                # Use the workflow if available
                state = {
                    **input_data,
                    "agent_outcome": None
                }
                result = await self.workflow.ainvoke(state)
                output = result.get("agent_outcome", {}).get("output", "")
                #print("output:" + str(output), end='\n')                
                if not output and isinstance(result.get("agent_outcome"), (str, dict)):
                    output = result["agent_outcome"].get("text", "") if isinstance(result["agent_outcome"], dict) else str(result["agent_outcome"])
            else:
                # Direct execution for different executor types
                if hasattr(self.agent_executor, 'ainvoke'):
                    # Handle any runnable with ainvoke (including RunnableSequence)
                    result = await self.agent_executor.ainvoke(input_data)
                    if isinstance(result, dict):
                        output = result.get("output", result.get("text", ""))
                    else:
                        output = str(result)
                else:
                    return self._format_response(
                        False, 
                        error=f"Unsupported agent_executor type: {type(self.agent_executor)}"
                    )
            
            return self._format_response(
                True,
                data={"response": output},
                metadata={"source": self.name}
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._format_response(False, error=str(e))
    
    def _format_response(
        self, 
        success: bool, 
        data: Optional[Dict[str, Any]] = None, 
        error: Optional[str] = None, 
        **metadata
    ) -> AgentResponse:
        """Helper method to format consistent responses"""
        return AgentResponse(
            success=success,
            data=data or {},
            error=error,
            metadata=metadata or {}
        )
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, tools={len(self.tools)})"

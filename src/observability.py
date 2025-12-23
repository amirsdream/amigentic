"""Observability setup using Arize Phoenix."""

import logging
from typing import Optional, Any

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .config import Config

logger = logging.getLogger(__name__)


class ObservabilityManager:
    """Manager for Phoenix observability setup."""

    def __init__(self, config: Config):
        """Initialize observability manager.
        
        Args:
            config: Application configuration.
        """
        self.config = config
        self.session: Optional[Any] = None
        self._instrumented = False

    def setup(self) -> bool:
        """Set up Phoenix observability.
        
        Returns:
            True if setup was successful, False otherwise.
        """
        try:
            # Launch Phoenix app
            self.session = px.launch_app()
            
            # Configure OpenTelemetry
            endpoint = f"http://127.0.0.1:{self.config.phoenix_port}/v1/traces"
            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
            trace_api.set_tracer_provider(tracer_provider)
            
            # Instrument LangChain
            LangChainInstrumentor().instrument()
            self._instrumented = True
            
            logger.info(f"Phoenix observability started at {self.get_url()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up observability: {e}")
            return False

    def get_url(self) -> str:
        """Get the Phoenix UI URL.
        
        Returns:
            The Phoenix URL or a default message.
        """
        if self.session and hasattr(self.session, 'url'):
            return self.session.url
        return f"http://localhost:{self.config.phoenix_port}"

    def is_active(self) -> bool:
        """Check if observability is active.
        
        Returns:
            True if observability is running, False otherwise.
        """
        return self.session is not None and self._instrumented

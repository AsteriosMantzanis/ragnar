from __future__ import annotations

import os
import uuid
from pathlib import Path

import requests
import streamlit as st


# Configuration

API_URL = os.getenv('API_URL', 'http://localhost:8000')

LOGO_PATH = Path(__file__).parent / 'assets' / 'ragnar-logo.png'

st.set_page_config(
    page_title='Ragnar',
    page_icon='◆',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Styling

st.markdown(
    """
    <style>
        ...existing styles...

        /* Sources */

        .source-card {
            padding: 0.9rem !important;
            margin: 0.5rem 0 !important;
            border: 1px solid var(--border) !important;
            border-left: 3px solid var(--primary) !important;
            border-radius: 8px !important;
            background: var(--surface-muted) !important;
        }

        .source-header {
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            gap: 1rem !important;
            margin-bottom: 0.4rem !important;
        }

        .source-title {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            color: var(--text) !important;
        }

        .source-score {
            font-size: 0.7rem !important;
            color: var(--muted) !important;
            white-space: nowrap !important;
        }

        .source-text {
            color: var(--text) !important;
            font-size: 0.85rem !important;
            line-height: 1.5 !important;
            word-break: break-word !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# API Client


class RagnarClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def query(
        self,
        query: str,
        session_id: str,
        strategy: str,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/query",
            json={
                'query': query,
                'session_id': session_id,
                'strategy': strategy,
            },
            timeout=120,
        )

        response.raise_for_status()

        return response.json()

    def get_session(self, session_id: str) -> dict:
        response = requests.get(
            f"{self.base_url}/sessions/{session_id}",
            timeout=10,
        )

        response.raise_for_status()

        return response.json()


@st.cache_resource
def get_client() -> RagnarClient:
    return RagnarClient(API_URL)


# Utilities


def new_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


def grounding_class(percentage: float) -> str:
    if percentage >= 75:
        return 'grounding-high'

    if percentage >= 50:
        return 'grounding-medium'

    return 'grounding-low'


def grounding_label(percentage: float) -> str:
    if percentage >= 75:
        return 'Grounded'

    if percentage >= 50:
        return 'Partially grounded'

    return 'Low grounding'


def render_grounding(percentage: float) -> None:
    css_class = grounding_class(percentage)
    label = grounding_label(percentage)

    st.markdown(
        f"""
        <div class="grounding {css_class}">
            {percentage:.0f}% · {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(
        f"Retrieved sources · {len(sources)}",
        expanded=False,
    ):
        for index, source in enumerate(sources[:8], start=1):
            text = source.get('text', '').strip()
            filename = source.get(
                'source',
                source.get('filename', 'Unknown source'),
            )

            score = source.get('rerank_score', source.get('score'))
            score_text = (
                f"{float(score):.3f}"
                if score is not None
                else '—'
            )

            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(f"**{index}. {filename}**")
            with col2:
                st.caption(f"score {score_text}")

            st.caption(text)
            st.divider()


def render_metrics(messages: list[dict]) -> None:
    assistant_messages = [
        message
        for message in messages
        if message['role'] == 'assistant'
    ]
    if not assistant_messages:
        return

    grounding_scores = [
        message.get('grounding_percentage', 0)
        for message in assistant_messages
    ]
    average_grounding = (
        sum(grounding_scores) / len(grounding_scores)
    )

    # Extract metrics if available
    response_times = [
        message.get('metrics', {}).get('total_duration_s', 0)
        for message in assistant_messages
        if message.get('metrics')
    ]
    avg_response_time = (
        sum(response_times) / len(response_times)
        if response_times else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Messages</div>
                <div class="metric-value">{len(messages)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Response</div>
                <div class="metric-value">{avg_response_time:.2f}s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Questions</div>
                <div class="metric-value">{len(assistant_messages)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Grounding</div>
                <div class="metric-value">{average_grounding:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_query_metrics(metrics: dict) -> None:
    """Render metrics for a single query."""

    if not metrics or not metrics.get('steps'):
        return

    with st.expander('Query performance'):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                'Response time',
                f"{metrics['total_duration_s']:.2f} s",
            )

        with col2:
            st.metric(
                'Sources',
                metrics['num_sources'],
            )

        with col3:
            st.metric(
                'Grounding',
                f"{metrics['grounding_percentage']:.0f}%",
            )

        st.divider()

        steps_data = [
            {
                'Step': step['name'].capitalize(),
                'Duration': f"{step['duration_s']:.2f} s",
            }
            for step in metrics['steps']
        ]

        st.dataframe(
            steps_data,
            use_container_width=True,
            hide_index=True,
        )

# Session State


def initialize_state() -> None:
    if 'client' not in st.session_state:
        st.session_state.client = get_client()

    if 'session_id' not in st.session_state:
        st.session_state.session_id = new_session_id()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'strategy' not in st.session_state:
        st.session_state.strategy = 'hierarchical'


# Sidebar

def render_sidebar() -> None:
    with st.sidebar:

        if LOGO_PATH.exists():
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.image(
                    str(LOGO_PATH),
                    width=150,
                )

        st.markdown(
            '<div class="sidebar-label">Session</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="session-id">
                {st.session_state.session_id}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write('')

        if st.button(
            '＋ New session',
            use_container_width=True,
        ):
            st.session_state.session_id = new_session_id()
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.markdown(
            '<div class="sidebar-label">Retrieval strategy</div>',
            unsafe_allow_html=True,
        )

        st.session_state.strategy = st.radio(
            'Retrieval strategy',
            options=['hierarchical', 'flat'],
            index=0,
            label_visibility='collapsed',
        )

        st.divider()

        st.markdown(
            '<div class="sidebar-label">System</div>',
            unsafe_allow_html=True,
        )

        st.caption(f"API · {API_URL}")
        st.caption('Ragnar RAG framework')


# Chat

def render_messages() -> None:
    for message in st.session_state.messages:
        role = message['role']

        with st.chat_message(role):
            st.markdown(message['content'])

            if role != 'assistant':
                continue

            # Grounding for this specific answer
            grounding_percentage = message.get(
                'grounding_percentage',
            )

            if grounding_percentage is not None:
                render_grounding(grounding_percentage)

            # Sources used for this specific answer
            render_sources(
                message.get('sources', []),
            )

            # Performance metrics for this specific query
            render_query_metrics(
                message.get('metrics', {}),
            )

            # cache bool and score
            render_cache_status(
                message.get('metrics', {}),
            )


def handle_query(query: str) -> None:
    client: RagnarClient = st.session_state.client
    session_id: str = st.session_state.session_id
    strategy: str = st.session_state.strategy

    st.session_state.messages.append(
        {
            'role': 'user',
            'content': query,
        },
    )

    try:
        with st.status(
            'Running Ragnar pipeline...',
            expanded=True,
        ) as status:
            st.write('Processing query...')

            result = client.query(
                query=query,
                session_id=session_id,
                strategy=strategy,
            )

            status.update(
                label='Completed',
                state='complete',
                expanded=False,
            )

        st.session_state.messages.append(
            {
                'role': 'assistant',
                'content': result.get(
                    'answer',
                    'No answer returned.',
                ),
                'grounding_percentage': result.get(
                    'grounded_percentage',
                    0,
                ),
                'sources': result.get(
                    'sources',
                    [],
                ),
                'metrics': result.get(
                    'metrics',
                    {},
                ),
            },
        )

    except requests.exceptions.Timeout:
        st.error(
            'The Ragnar API timed out while processing '
            'the request.',
        )

    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to Ragnar API at {API_URL}.",
        )

    except requests.exceptions.HTTPError as exc:
        st.error(
            f"Ragnar API returned an error: {exc}",
        )

    except Exception as exc:
        st.error(
            f"Unexpected error: {exc}",
        )

# cache


def render_cache_status(metrics: dict) -> None:
    """Show if answer was cached or generated."""
    if not metrics:
        return

    cache_hit = metrics.get('cache_hit', False)
    cache_score = metrics.get('cache_score')

    if cache_hit:
        score_text = f" · {cache_score:.2f}" if cache_score else ''
        st.markdown(
            f"""
            <div style="display: inline-block;
            padding: 0.4rem 0.8rem;
            margin-top: 0.5rem;
            border-radius: 6px;
            background: #d4edda;
            color: #155724;
            font-size: 0.85rem;">
                📦 From Cache{score_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="display: inline-block;
            padding: 0.4rem 0.8rem;
            margin-top: 0.5rem;
            border-radius: 6px;
            background: #e7e8ea;
            color: #383d41;
            font-size: 0.85rem;">
                ◉ Generated
            </div>
            """,
            unsafe_allow_html=True,
        )

# Main


def main() -> None:
    initialize_state()

    render_sidebar()

    st.markdown(
        """
        <div class="ragnar-header">
            <div class="ragnar-title">Ragnar</div>
            <div class="ragnar-subtitle">
                Retrieval-augmented generation workbench
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.info(
            'Ask a question about your indexed documents '
            'to start a conversation.',
        )

    render_messages()

    query = st.chat_input(
        'Ask Ragnar something...',
    )

    if query:
        handle_query(query)
        st.rerun()

    if st.session_state.messages:
        st.divider()
        render_metrics(
            st.session_state.messages,
        )


if __name__ == '__main__':
    main()

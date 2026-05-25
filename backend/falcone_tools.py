#--------------------------------------------------------------------------------------------------
# MODULE IMPORTS
#--------------------------------------------------------------------------------------------------
# The tool decorator from LangChain wraps a normal Python function so the model can call it.
# It auto-generates the JSON schema from the function signature and docstring, which is what
# the model sees when deciding whether/how to call the tool. The docstring IS the description
# the model uses, so write it for the model, not just for humans.
from langchain_core.tools import tool

# DDGS is the DuckDuckGo search client. The library was previously named "duckduckgo-search"
# but has been renamed to "ddgs". No API key required, which is perfect for the lab.
from ddgs import DDGS

# BeautifulSoup strips HTML down to readable text so we don't dump raw markup into the prompt.
from bs4 import BeautifulSoup

# We use requests for URL fetching to stay consistent with the rest of the codebase.
import requests

# urlparse lets us pull apart a URL into scheme, hostname, path, etc. for validation.
# ipaddress and socket are used for the baseline SSRF check.
from urllib.parse import urlparse
import ipaddress
import socket
#--------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------
# CONSTANTS
#--------------------------------------------------------------------------------------------------
# Keep all limits visible in one place so they are easy to find when red-teaming.
# These are baseline safety values — they are deliberately not bulletproof.

# Max number of search results returned to the model. More results = more attack surface
# (every result is untrusted text the model will read).
WEB_SEARCH_MAX_RESULTS = 5

# Hard timeout on URL fetch. Without this, a slow-loris-style endpoint could hang the request.
URL_FETCH_TIMEOUT_SECONDS = 10

# Byte ceiling on the raw response. Stops a hostile page from blowing up memory or token usage.
URL_FETCH_MAX_BYTES = 200_000

# Final character cap after HTML parsing. Even after stripping tags, we truncate before handing
# text to the model — protects context window from being flooded (LLM04 DoS surface).
URL_FETCH_MAX_CHARS = 5_000

# Only allow http/https. Blocks file://, ftp://, gopher://, etc.
ALLOWED_SCHEMES = {"http", "https"}
#--------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------
# SSRF HELPER
#--------------------------------------------------------------------------------------------------
# This function resolves a hostname to an IP address and rejects private/loopback/link-local
# ranges. The goal is to stop the model from being tricked into fetching internal endpoints
# (e.g., http://localhost:8000, http://169.254.169.254 for cloud metadata, http://10.0.0.1).
#
# DELIBERATE GAPS LEFT FOR LAB ATTACK PRACTICE (LLM07 Insecure Plugin Design):
# - DNS rebinding: we resolve once here, but requests resolves again at fetch time. An attacker
#   controlling DNS can return a public IP on the first lookup and a private IP on the second.
# - IPv6 edge cases (::ffff:127.0.0.1 mapped addresses, etc.) are not normalized.
# - Open redirects on allowed hosts can still bounce the bot to anywhere.
# - No domain allowlist — any public host is fair game.
def _is_private_address(hostname: str) -> bool:
    try:
        # Resolve the hostname to an IP string and parse it.
        resolved_ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(resolved_ip)
        # Reject anything that should never be reachable from a user-controlled URL.
        return (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        )
    # If we cannot resolve, fail closed — refuse to fetch.
    except (socket.gaierror, ValueError):
        return True
#--------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------
# WEB SEARCH TOOL
#--------------------------------------------------------------------------------------------------
# The @tool decorator turns this function into a LangChain Tool object. The model sees:
#   - Tool name: "web_search" (from the function name)
#   - Description: from the docstring (write this carefully — model uses it to decide when to call)
#   - Argument schema: derived from type hints ({"query": str})
#
# When the model calls this, LangChain validates the arguments against the schema and invokes
# the function. Whatever string this returns goes back to the model as a ToolMessage.
@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo.

    Use this only when the user asks about current events, recent news, or external
    facts not covered by your training data or the uploaded family documents.
    Do not use for greetings, small talk, or questions you can already answer.

    Args:
        query: A short search phrase, for example "Gotham mayor 2025".

    Returns:
        Top search results formatted as numbered entries with title, URL, and snippet.
    """
    # Defensive check — the model can pass anything as a string, including empty ones.
    if not query or not query.strip():
        return "Web search error: empty query."
    
    try:
        # The context manager (with DDGS() as ddgs) handles cleanup of the HTTP session.
        with DDGS() as ddgs:
            # .text() returns a list of dicts with keys: title, href, body.
            # We cap results so a single tool call cannot flood the context window.
            raw_results = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS))
    except Exception as error:
        # Any failure (network, rate limit, parse error) is returned as a tool error string.
        # We do NOT raise — the model handles the string and decides what to do.
        return f"Web search error: {error}"
    
    if not raw_results:
        return f"No web results found for query: {query}"
    
    # Format results into a readable block. Note: the snippet field comes from the open web —
    # treat every character of it as attacker-controllable input (LLM01 indirect injection).
    formatted_results = []
    for index, result in enumerate(raw_results, start=1):
        title = result.get("title", "(no title)")
        url = result.get("href", "(no url)")
        snippet = result.get("body", "(no snippet)")
        formatted_results.append(
            f"[{index}] {title}\n    URL: {url}\n    Snippet: {snippet}"
        )
    
    return "\n\n".join(formatted_results)
#--------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------
# URL FETCH TOOL
#--------------------------------------------------------------------------------------------------
# Fetches a URL, parses out readable text, and returns it. This is the highest-risk tool —
# it is the primary indirect prompt injection vector and the SSRF vector simultaneously.
@tool
def url_fetch(url: str) -> str:
    """Fetch the readable text of a web page.

    Use this when the user provides a URL or when a web_search result looks promising
    and you need the full page contents to answer accurately. Do not call for arbitrary URLs
    that were not provided by the user or surfaced by web_search.

    Args:
        url: A complete URL starting with http:// or https://.

    Returns:
        Extracted readable text of the page, truncated to a safe length.
    """
    if not url or not url.strip():
        return "URL fetch error: empty URL."
    
    # urlparse breaks the URL into components so we can validate each piece independently.
    parsed = urlparse(url.strip())
    
    # Scheme check. file://, ftp://, gopher:// etc. are rejected here.
    if parsed.scheme not in ALLOWED_SCHEMES:
        return f"URL fetch error: scheme '{parsed.scheme}' is not allowed. Use http or https."
    
    # A URL like "http:///path" with no hostname is invalid and would otherwise cause
    # requests to throw on its own — we catch it earlier with a clearer message.
    if not parsed.hostname:
        return "URL fetch error: no hostname in URL."
    
    # Baseline SSRF guard. See _is_private_address comments for known gaps you can exploit.
    if _is_private_address(parsed.hostname):
        return f"URL fetch error: refusing to fetch private/internal address {parsed.hostname}."
    
    try:
        # stream=True with raw.read(N) gives us a hard cap on bytes downloaded.
        # Without stream=True, requests would download the entire response into memory first.
        with requests.get(
            url,
            timeout=URL_FETCH_TIMEOUT_SECONDS,
            stream=True,
            headers={"User-Agent": "Falcone-Bot/1.0 (lab)"},
            # allow_redirects=True is the default. Note: redirects are NOT re-validated against
            # the SSRF check — an attacker can host a public page that redirects to localhost.
            # Left in place as a deliberate attack surface.
        ) as response:
            response.raise_for_status()
            # Read at most MAX_BYTES + 1 so we can detect if we hit the cap.
            raw_bytes = response.raw.read(URL_FETCH_MAX_BYTES + 1, decode_content=True)
    except Exception as error:
        return f"URL fetch error: {error}"
    
    truncated_notice = ""
    if len(raw_bytes) > URL_FETCH_MAX_BYTES:
        # Trim back to the cap and tell the model we cut it off.
        raw_bytes = raw_bytes[:URL_FETCH_MAX_BYTES]
        truncated_notice = f"\n\n[Page truncated at {URL_FETCH_MAX_BYTES} bytes.]"
    
    # Decode bytes to text. errors="replace" prevents one bad byte from blowing up the call.
    html_text = raw_bytes.decode("utf-8", errors="replace")
    
    # BeautifulSoup parses HTML. We use the built-in html.parser so we do not need lxml.
    soup = BeautifulSoup(html_text, "html.parser")
    
    # Strip executable/styling/noscript tags before extracting text. Their inner text is
    # almost never useful to the model and frequently contains junk that wastes tokens.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    
    # get_text(separator="\n", strip=True) flattens the DOM to text with line breaks between
    # elements, which preserves some readability without dumping HTML.
    extracted_text = soup.get_text(separator="\n", strip=True)
    
    # Final character cap. Even after HTML stripping, large pages will swamp the context.
    if len(extracted_text) > URL_FETCH_MAX_CHARS:
        extracted_text = extracted_text[:URL_FETCH_MAX_CHARS] + "\n\n[Text truncated.]"
    
    # Return the URL alongside the content so the model can cite/reference it.
    return f"Fetched URL: {url}\n\n{extracted_text}{truncated_notice}"
#--------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------
# TOOL REGISTRY
#--------------------------------------------------------------------------------------------------
# One list for binding to the LLM via .bind_tools(FALCONE_TOOLS).
FALCONE_TOOLS = [web_search, url_fetch]

# Name-to-tool map used by the tool execution loop in falcone_chain.py to dispatch calls.
# We use t.name (not the function __name__) because @tool can override the name.
FALCONE_TOOLS_BY_NAME = {t.name: t for t in FALCONE_TOOLS}
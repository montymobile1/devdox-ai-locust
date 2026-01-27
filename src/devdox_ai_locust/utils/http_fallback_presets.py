from typing import Dict, Union, Iterable, Optional, Set
from pydantic import BaseModel
import json
import yaml


class ResponseBlock(BaseModel):
    """
    A Pydantic model that wraps OpenAPI-style response definitions,
    with export capabilities to JSON and YAML.
    """

    responses: Dict[str, Dict[str, Dict[str, str]]]

    def to_json(self, **kwargs) -> str:
        """Export the responses to a JSON-formatted string."""
        return json.dumps(self.responses, **kwargs)

    def to_yaml(self) -> str:
        """Export the responses to a YAML-formatted string."""
        return yaml.dump(self.responses, sort_keys=False)

    def as_dict(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Return the raw dictionary representation of the responses."""
        return self.responses


class FallbackHttpResponseRegistry:
    """
    Provides fallback OpenAPI-style HTTP response definitions with deterministic,
    LLM-safe descriptions. 5xx COMMON responses are automatically merged per method.
    """

    _COMMON: Dict[str, Dict[str, str]] = {
        "500": {
            "description": "The server encountered an internal error during request processing."
        },
        "502": {
            "description": "The server received an invalid response from an upstream server."
        },
        "503": {
            "description": "The server was unavailable or overloaded and could not handle the request."
        },
        "504": {
            "description": "The server timed out while waiting for a response from an upstream server."
        },
    }

    _BASE_RESPONSES: Dict[str, Dict[str, Dict[str, str]]] = {
        "GET": {
            "200": {
                "description": "The server accepted the request and returned a response body."
            },
            "204": {
                "description": "The server accepted the request and returned no response body."
            },
            "400": {
                "description": "The server rejected the request before executing it."
            },
            "401": {
                "description": "The server rejected the request because authentication was not accepted."
            },
            "403": {
                "description": "The server accepted authentication but rejected execution of the request."
            },
            "404": {
                "description": "The server could not match the request to any available handler."
            },
            "422": {
                "description": "The server parsed the request but rejected it before execution."
            },
        },
        "POST": {
            "201": {
                "description": "The server executed the request and created new server-side state."
            },
            "200": {
                "description": "The server executed the request and returned a response body."
            },
            "204": {
                "description": "The server executed the request and returned no response body."
            },
            "400": {"description": "The server rejected the request before execution."},
            "401": {
                "description": "The server rejected the request because authentication was not accepted."
            },
            "403": {
                "description": "The server accepted authentication but rejected execution of the request."
            },
            "409": {
                "description": "The server rejected the request due to a conflict with existing server-side state."
            },
            "415": {
                "description": "The server rejected the request before execution due to unsupported input format."
            },
            "422": {
                "description": "The server parsed the request but rejected it before execution."
            },
        },
        "PUT": {
            "200": {
                "description": "The server executed the request and replaced existing server-side state."
            },
            "204": {
                "description": "The server executed the request and returned no response body."
            },
            "201": {
                "description": "The server executed the request and created new server-side state."
            },
            "400": {"description": "The server rejected the request before execution."},
            "401": {
                "description": "The server rejected the request because authentication was not accepted."
            },
            "403": {
                "description": "The server accepted authentication but rejected execution of the request."
            },
            "409": {
                "description": "The server rejected the request due to a conflict with existing server-side state."
            },
            "422": {
                "description": "The server parsed the request but rejected it before execution."
            },
        },
        "PATCH": {
            "200": {
                "description": "The server executed the request and updated server-side state."
            },
            "204": {
                "description": "The server executed the request and returned no response body."
            },
            "400": {"description": "The server rejected the request before execution."},
            "401": {
                "description": "The server rejected the request because authentication was not accepted."
            },
            "403": {
                "description": "The server accepted authentication but rejected execution of the request."
            },
            "409": {
                "description": "The server rejected the request due to a conflict with existing server-side state."
            },
            "415": {
                "description": "The server rejected the request before execution due to unsupported input format."
            },
            "422": {
                "description": "The server parsed the request but rejected it before execution."
            },
        },
        "DELETE": {
            "204": {
                "description": "The server executed the request and returned no response body."
            },
            "200": {
                "description": "The server executed the request and returned a response body."
            },
            "404": {
                "description": "The server could not match the request to any available handler."
            },
            "401": {
                "description": "The server rejected the request because authentication was not accepted."
            },
            "403": {
                "description": "The server accepted authentication but rejected execution of the request."
            },
            "422": {
                "description": "The server parsed the request but rejected it before execution."
            },
        },
    }

    _AUTH_CODES = {"401", "403"}

    def __init__(self):
        """Initialize with merged COMMON codes into each method."""
        self._RESPONSES = {
            method: {**codes, **self._COMMON}
            for method, codes in self._BASE_RESPONSES.items()
        }

    def get_responses(
        self,
        methods: Union[str, Iterable[str]],
        status: Optional[Iterable[Union[str, int]]] = None,
        exclude_status: Optional[Iterable[Union[str, int]]] = None,
        exclude_auth: bool = False,
    ) -> ResponseBlock:
        """
        Get fallback response definitions per method, with filtering options.

        Parameters
        ----------
        methods : str or list of str
            HTTP methods (e.g. "GET", ["GET", "POST"]).
        status : list of str or int, optional
            Filter by exact code or class ("404", "2xx", "5xx").
        exclude_status : list of str or int, optional
            Codes or classes to exclude from result.
        exclude_auth : bool
            If True, removes 401 and 403.

        Returns
        -------
        ResponseBlock : with .to_json(), .to_yaml(), .as_dict()
        """
        methods = {methods} if isinstance(methods, str) else set(methods)
        requested_status = self._expand_status_selectors(status)
        excluded_status = self._expand_status_selectors(exclude_status)

        result: Dict[str, Dict[str, Dict[str, str]]] = {}

        for method in methods:
            method = method.upper()
            responses = dict(self._RESPONSES.get(method, {}))

            if requested_status:
                responses = {
                    code: data
                    for code, data in responses.items()
                    if code in requested_status
                }

            if excluded_status:
                responses = {
                    code: data
                    for code, data in responses.items()
                    if code not in excluded_status
                }

            if exclude_auth:
                responses = {
                    code: data
                    for code, data in responses.items()
                    if code not in self._AUTH_CODES
                }

            if responses:
                result[method] = responses

        return ResponseBlock(responses=result)

    @staticmethod
    def _expand_status_selectors(
        selectors: Optional[Iterable[Union[str, int]]],
    ) -> Set[str]:
        """Expand status selectors like '4xx' or 500 into concrete strings."""
        if not selectors:
            return set()

        expanded: Set[str] = set()

        for sel in selectors:
            sel = str(sel)
            if sel.endswith("xx") and len(sel) == 3:
                try:
                    prefix = int(sel[0])
                    expanded.update(
                        {str(code) for code in range(prefix * 100, (prefix + 1) * 100)}
                    )
                except (ValueError, IndexError):
                    expanded.add(sel)
                    continue
            else:
                expanded.add(sel)

        return expanded

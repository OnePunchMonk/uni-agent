from __future__ import annotations

import pytest

from uni_agent.gateway.session.codec import _flatten_image_url_blocks


@pytest.mark.cpu
@pytest.mark.level0
def test_flatten_image_url_blocks_unwraps_openai_url_dict_without_mutating_input():
    messages = [
        {"role": "system", "content": "plain text"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "what is this?"},
            ],
        },
    ]

    flattened = _flatten_image_url_blocks(messages)

    assert flattened[0] is messages[0]
    assert flattened[1]["content"][0] == {"type": "image_url", "image_url": "data:image/png;base64,AAAA"}
    assert flattened[1]["content"][1] == {"type": "text", "text": "what is this?"}
    assert messages[1]["content"][0]["image_url"] == {"url": "data:image/png;base64,AAAA"}


@pytest.mark.cpu
@pytest.mark.level0
def test_flatten_image_url_blocks_leaves_string_urls_and_other_blocks_alone():
    content = [
        {"type": "image_url", "image_url": "https://example.com/a.png"},
        {"type": "image", "image": "file:///tmp/a.png"},
        {"type": "image_url", "image_url": {}},
    ]

    flattened = _flatten_image_url_blocks([{"role": "user", "content": content}])

    assert flattened[0]["content"] == content

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


class _SpyBuilder:
    """Stands in for the Continuous Token builder to observe what the codec renders."""

    def __init__(self, *, multimodal: bool = True):
        self.multimodal = multimodal
        self.renders: list[dict] = []
        self.merges: list[tuple] = []

    def supports_multimodal(self) -> bool:
        return self.multimodal

    def render_tokens_with_mm(self, messages, images, *, videos=None, add_generation_prompt, tools=None):
        self.renders.append(
            {"n_messages": len(messages), "images": list(images), "videos": videos, "gen": add_generation_prompt}
        )
        # One numbered token per message / image / video, so a shorter render is a prefix of a longer one.
        count = len(messages) + len(images) + len(videos or [])
        return [100 + i for i in range(count)] + ([900] if add_generation_prompt else [])

    def _merge_context_token_ids(self, runtime_token_ids, appended_token_ids, **kwargs):
        self.merges.append((list(runtime_token_ids), list(appended_token_ids), kwargs))
        from types import SimpleNamespace

        return SimpleNamespace(token_ids=list(runtime_token_ids) + list(appended_token_ids), kind="context")

    def align_response_metadata(self, merge_result, response_mask, response_logprobs):
        return list(response_mask) + [0] * 3, response_logprobs


def _codec_with(builder):
    from tests.uni_agent.support import FakeTokenizer
    from uni_agent.gateway.session.codec import MessageCodec

    codec = MessageCodec(FakeTokenizer())
    codec._continuous_token_builder = builder
    return codec


@pytest.mark.cpu
@pytest.mark.level0
def test_incremental_media_is_rendered_with_previous_and_new_processed_media():
    builder = _SpyBuilder()
    codec = _codec_with(builder)
    previous = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    updated = previous + [{"role": "user", "content": "c"}]

    token_ids, mask, _ = codec.merge_context_tokens(
        previous,
        updated,
        [1, 1, 7],
        [0],
        image_data=["new-img"],
        video_data=["new-vid"],
        previous_image_data=["old-img"],
        previous_video_data=["old-vid"],
    )

    prefix_render, full_render = builder.renders
    assert (
        prefix_render["images"] == ["old-img"] and prefix_render["videos"] == ["old-vid"] and not prefix_render["gen"]
    )
    assert full_render["images"] == ["old-img", "new-img"] and full_render["videos"] == ["old-vid", "new-vid"]
    assert full_render["gen"]
    runtime, appended, kwargs = builder.merges[0]
    # prefix = 2 messages + 1 image + 1 video = 4 tokens; full = 3 + 2 + 2 = 7 tokens plus the generation prompt
    assert appended == [104, 105, 106, 900]
    assert kwargs["appended_messages"] == updated[2:]
    assert token_ids == runtime + appended and len(mask) == 4


@pytest.mark.cpu
@pytest.mark.level0
def test_incremental_media_rejects_non_multimodal_builder():
    codec = _codec_with(_SpyBuilder(multimodal=False))

    with pytest.raises(ValueError, match="not a multimodal builder"):
        codec.merge_context_tokens([], [], [1], [0], image_data=["img"])


@pytest.mark.cpu
@pytest.mark.level0
def test_incremental_media_rejects_a_prefix_that_is_not_a_token_prefix():
    class _Unstable(_SpyBuilder):
        def render_tokens_with_mm(self, messages, images, *, videos=None, add_generation_prompt, tools=None):
            return [5] if not add_generation_prompt else [6, 6]

    with pytest.raises(ValueError, match="suffix diff failed"):
        _codec_with(_Unstable()).merge_context_tokens([], [], [1], [0], image_data=["img"])

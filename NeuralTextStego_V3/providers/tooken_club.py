"""Провайдер Tooken Club."""

from providers.openai_compatible import OpenAICompatibleProvider


class TookenClubProvider(OpenAICompatibleProvider):
    name = "tooken_club"
    label = "Tooken Club"
    API_BASE = "https://tooken.club/v1"
    api_key_setting = "tooken_club_api_key"
    model_setting = "tooken_club_model"

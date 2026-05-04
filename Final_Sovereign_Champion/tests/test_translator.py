import pytest
import sys
import os

# Add the parent directory to sys.path so we can import translator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translator import dict_translate, env_translate, dict_help_screens

def test_dict_translate_de():
    res = dict_translate("de")
    assert isinstance(res, dict)
    assert res["HelpMode"] == "Hilfemodus"
    assert "Round" in res

def test_dict_translate_en():
    res = dict_translate("en")
    assert isinstance(res, dict)
    assert res["HelpMode"] == "Help Mode"
    assert "Round" in res

def test_dict_translate_invalid():
    with pytest.raises(RuntimeError) as excinfo:
        dict_translate("fr")
    assert "language = fr is not supported" in str(excinfo.value)

def test_env_translate_de():
    res = env_translate("de")
    assert isinstance(res, dict)
    assert res["NumAPointsTooLow"] == "Zu wenig Aktionspunkte"

def test_env_translate_en():
    res = env_translate("en")
    assert isinstance(res, dict)
    assert res["NumAPointsTooLow"] == "Too few action points"

def test_env_translate_invalid():
    with pytest.raises(RuntimeError) as excinfo:
        env_translate("es")
    assert "language = es is not supported" in str(excinfo.value)

def test_dict_help_screens_de():
    res = dict_help_screens("de")
    assert isinstance(res, dict)
    assert "Willkommen bei Ökolopoly!" in res["hs1.0"]

def test_dict_help_screens_en():
    res = dict_help_screens("en")
    assert isinstance(res, dict)
    assert "Welcome to Ökolopoly!" in res["hs1.0"]

def test_dict_help_screens_invalid():
    with pytest.raises(RuntimeError) as excinfo:
        dict_help_screens("it")
    assert "language = it is not supported" in str(excinfo.value)

"""GPU assignment, with stand-in GPUs (no graphics card needed)."""
import json

import pytest

from avatarstream import cli
from avatarstream import gpus as G

STRONG = G.GPU(0, "RTX 4090", "GPU-aaaa", 24.0, 128)
WEAK = G.GPU(1, "RTX 3060", "GPU-bbbb", 12.0, 28)


@pytest.fixture
def two_gpus(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    cards = [G.GPU(**vars(STRONG)), G.GPU(**vars(WEAK))]
    monkeypatch.setattr(G, "list_gpus", lambda: cards)
    monkeypatch.setattr(G, "benchmark", lambda g, seconds=0.6: setattr(g, "tflops", 160.0 if g.uuid == "GPU-aaaa" else 50.0))
    return cards


def _answers(monkeypatch, *replies):
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


@pytest.mark.parametrize("kinds, lipsync_on", [
    ({"lipsync": "wav2lip", "voice_changer": "rvc"}, "GPU-aaaa"),      # tie: lip sync has a hard time budget
    ({"lipsync": "wav2lip", "voice_changer": "seed-vc"}, "GPU-bbbb"),  # Seed-VC is the heavier task
    ({"lipsync": "musetalk", "voice_changer": "rvc"}, "GPU-aaaa"),     # MuseTalk is the heavier task, RVC is light
    ({"lipsync": "musetalk", "voice_changer": "seed-vc"}, "GPU-aaaa"),  # tie: lip sync has a hard time budget
])
def test_heaviest_task_gets_the_fastest_card(kinds, lipsync_on):
    plan = G.recommend([WEAK, STRONG], kinds)
    assert plan["lipsync"].uuid == lipsync_on
    assert plan["voice_changer"].uuid != lipsync_on


def test_single_card_and_no_card():
    plan = G.recommend([STRONG], {"lipsync": "wav2lip", "voice_changer": "rvc"})
    assert plan["lipsync"] is plan["voice_changer"] is STRONG
    assert G.recommend([], {}) == {"lipsync": None, "voice_changer": None}


def test_measured_speed_beats_core_count():
    old_big = G.GPU(0, "old big card", "GPU-old", 24.0, 80, tflops=30.0)
    new_small = G.GPU(1, "new small card", "GPU-new", 16.0, 60, tflops=90.0)
    plan = G.recommend([old_big, new_small], {"lipsync": "musetalk", "voice_changer": "rvc"})
    assert plan["lipsync"] is new_small


def test_config_picks_the_device_by_uuid_even_if_numbering_changes(tmp_path, monkeypatch):
    path = tmp_path / "gpu_config.json"
    G.save_config({"lipsync": WEAK, "voice_changer": STRONG}, {"lipsync": "wav2lip", "voice_changer": "rvc"}, str(path))
    renumbered = [G.GPU(0, "RTX 3060", "GPU-bbbb", 12.0, 28), G.GPU(1, "RTX 4090", "GPU-aaaa", 24.0, 128)]
    assert G.resolve_device("lipsync", "auto", renumbered, str(path)) == "cuda:0"
    assert G.resolve_device("voice_changer", None, renumbered, str(path)) == "cuda:1"
    assert G.resolve_device("lipsync", "cuda:1", renumbered, str(path)) == "cuda:1"  # explicit --device wins
    assert G.resolve_device("lipsync", "auto", [], str(path)) == "auto"  # card gone: fall back
    assert G.resolve_device("lipsync", "auto", renumbered, str(tmp_path / "none.json")) == "auto"


def test_interactive_setup_lets_the_user_override(two_gpus, monkeypatch, tmp_path, capsys):
    rvc = tmp_path / "RVC" / "go-realtime-gui.bat"
    rvc.parent.mkdir()
    rvc.write_text("@echo off\n")
    # RVC; lip sync: accept (strong card 0); voice changer: override the weak card with 0; launcher path.
    _answers(monkeypatch, "1", "", "0", str(rvc))
    cli.main(["gpus", "--setup"])
    out = capsys.readouterr().out
    assert "推荐方案" in out and "同一张卡" in out

    cfg = json.loads((tmp_path / "gpu_config.json").read_text(encoding="utf-8"))
    assert cfg["kinds"]["voice_changer"] == "rvc"
    assert cfg["tasks"]["lipsync"]["uuid"] == "GPU-aaaa"
    assert cfg["tasks"]["voice_changer"]["uuid"] == "GPU-aaaa"  # overridden by the user

    bat = (tmp_path / "scripts" / "start_voice_changer.bat").read_bytes().decode("utf-8")
    assert "set CUDA_VISIBLE_DEVICES=GPU-aaaa\r\n" in bat
    assert "set CUDA_DEVICE_ORDER=PCI_BUS_ID\r\n" in bat
    assert f'call "{rvc}"' in bat


def test_default_plan_puts_rvc_on_the_weaker_card(two_gpus, tmp_path):
    cli.main(["gpus", "--auto", "--voice-changer", "rvc"])
    cfg = json.loads((tmp_path / "gpu_config.json").read_text(encoding="utf-8"))
    assert (cfg["tasks"]["lipsync"]["uuid"], cfg["tasks"]["voice_changer"]["uuid"]) == ("GPU-aaaa", "GPU-bbbb")


def test_assign_and_auto(two_gpus, tmp_path):
    cli.main(["gpus", "--assign", "lipsync=0", "voice_changer=1", "--voice-changer", "seed-vc"])
    cfg = json.loads((tmp_path / "gpu_config.json").read_text(encoding="utf-8"))
    assert (cfg["tasks"]["lipsync"]["uuid"], cfg["tasks"]["voice_changer"]["uuid"]) == ("GPU-aaaa", "GPU-bbbb")

    cli.main(["gpus", "--auto"])  # keeps seed-vc from the saved config
    cfg = json.loads((tmp_path / "gpu_config.json").read_text(encoding="utf-8"))
    assert cfg["kinds"]["voice_changer"] == "seed-vc"
    assert (cfg["tasks"]["lipsync"]["uuid"], cfg["tasks"]["voice_changer"]["uuid"]) == ("GPU-bbbb", "GPU-aaaa")


def test_measured_lipsync_load_moves_it_off_a_card_that_cannot_keep_up(two_gpus, tmp_path, monkeypatch, capsys):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "wav2lip.pth").write_bytes(b"")
    monkeypatch.setattr(cli, "_test_lipsync", lambda gpus, m, b: {"GPU-aaaa": 0.2, "GPU-bbbb": 0.9})
    # Seed-VC is heavy, so the plan first puts lip sync on the weak card, where it cannot keep up.
    cli.main(["gpus", "--auto", "--test-lipsync", "--voice-changer", "seed-vc"])
    cfg = json.loads((tmp_path / "gpu_config.json").read_text(encoding="utf-8"))
    assert cfg["tasks"]["lipsync"]["uuid"] == "GPU-aaaa"
    assert cfg["tasks"]["voice_changer"]["uuid"] == "GPU-bbbb"
    assert "太吃力" in capsys.readouterr().out


def test_bad_assign_is_explained(two_gpus):
    with pytest.raises(SystemExit) as e:
        cli.main(["gpus", "--assign", "lipsync=7"])
    assert "任务=编号" in str(e.value)


def test_python_launcher_is_refused(tmp_path):
    with pytest.raises(ValueError):
        G.write_launcher(str(tmp_path / "real-time-gui.py"), STRONG, str(tmp_path / "x.bat"))

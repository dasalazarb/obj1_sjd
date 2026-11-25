"""Verify local inputs and deployment identity without loading a GPU model."""
from .inference import force_offline
from .resources import verify
from .run_pipeline import build_identity, inputs_for
from .settings import digest, load_config


def main() -> None:
    force_offline()
    try:
        cfg = load_config()
        identity = build_identity(cfg, verify(cfg), inputs_for(cfg))
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Semantic preflight failed ({type(error).__name__}).") from None
    print(f"Semantic preflight passed. Run identity {digest(identity)}")


if __name__ == "__main__":
    main()

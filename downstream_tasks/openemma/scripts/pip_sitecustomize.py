import importlib
import os
import sys
import time
import urllib.request


def get_sts_token():
    cached = os.getenv("AIEXT_REPO_AUTH_STS_TOKEN")
    if cached:
        return cached

    credentials_uri = os.getenv("ALIBABA_CLOUD_CREDENTIALS_URI")
    if not credentials_uri:
        return None

    last_error = None
    for _ in range(5):
        try:
            with urllib.request.urlopen(credentials_uri, timeout=5) as response:
                token = response.read().decode("utf-8")
                os.environ["AIEXT_REPO_AUTH_STS_TOKEN"] = token
                return token
        except Exception as exc:
            last_error = exc
            time.sleep(1)

    print(f"aiext pip auth warning: {last_error}")
    return None


if sys.argv[0].endswith("pip") or sys.argv[0].endswith("pip3"):
    import pip
    from pip._internal.network.session import PipSession

    class CustomPipSession(PipSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sts_token = get_sts_token()
            if sts_token:
                self.headers["repo-auth-sts-token"] = sts_token

    if os.getenv("ALIBABA_CLOUD_CREDENTIALS_URI") is not None:
        print("local aiext pip auth plugin has been loaded.")
        pip._internal.network.session.PipSession = CustomPipSession
        importlib.reload(pip)

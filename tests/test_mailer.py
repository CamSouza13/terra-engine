"""Customer email tests.

We never send real mail: SMTP is mocked to capture the composed message, so we can
assert the right recipient, subject, and both text+HTML parts. Also checks the
no-op behavior when SMTP is unconfigured, and that ordering/support create paths
attempt a customer confirmation.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout=10):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        pass

    def login(self, u, p):
        pass

    def send_message(self, m):
        _FakeSMTP.sent.append(m)


def _configure(monkeyenv=True):
    import smtplib
    from terra import mailer
    importlib.reload(mailer)
    _FakeSMTP.sent = []
    smtplib.SMTP = _FakeSMTP
    if monkeyenv:
        os.environ["TERRA_SMTP_HOST"] = "smtp.test"
        os.environ["TERRA_SMTP_FROM"] = "hello@terra.test"
    else:
        os.environ.pop("TERRA_SMTP_HOST", None)
    return mailer


def test_noop_without_smtp():
    m = _configure(monkeyenv=False)
    assert m.configured() is False
    assert m.send_welcome("cust@x.io", "Acme", 14) is False
    assert _FakeSMTP.sent == []
    print("  mailer: no-op when SMTP unconfigured OK")


def test_welcome_order_ticket_composed():
    m = _configure()
    assert m.send_welcome("cust@x.io", "Acme", 14, origin="https://terra.test") is True
    assert m.send_order_confirmation("buyer@x.io", 7, "Raspberry Pi 4 (4 GB)", "Field / mid-grade", 2) is True
    assert m.send_ticket_confirmation("help@x.io", 3, "Node offline") is True
    subs = [(msg["To"], msg["Subject"]) for msg in _FakeSMTP.sent]
    assert ("cust@x.io", "Welcome to Terra — your trial has started") in subs
    assert ("buyer@x.io", "Terra order #7 received") in subs
    assert ("help@x.io", "Terra support #3 received") in subs
    # multipart: text + html alternative present
    welcome = _FakeSMTP.sent[0]
    assert welcome.get_body(preferencelist=("html",)) is not None
    assert welcome.get_body(preferencelist=("plain",)) is not None
    assert "Acme" in welcome.get_body(preferencelist=("plain",)).get_content()
    print(f"  mailer: composed {len(_FakeSMTP.sent)} customer emails with text+HTML OK")


def test_order_path_sends_customer_confirmation(tmp_path=None):
    import tempfile
    m = _configure()
    with tempfile.TemporaryDirectory() as tmp:
        from terra import accounts, support
        importlib.reload(accounts)
        accounts.HOME = tmp
        accounts.DB_PATH = os.path.join(tmp, "terra.db")
        importlib.reload(support)
        before = len(_FakeSMTP.sent)
        support.create_order("buyer@x.io", "Raspberry Pi 4 (4 GB)", "Field / mid-grade", 1)
        support.create_ticket("buyer@x.io", "Help", "stuck")
        tos = [msg["To"] for msg in _FakeSMTP.sent[before:]]
        assert tos.count("buyer@x.io") >= 2   # order + ticket confirmations
    print("  mailer: order/support create paths email the customer OK")


if __name__ == "__main__":
    test_noop_without_smtp()
    test_welcome_order_ticket_composed()
    test_order_path_sends_customer_confirmation()
    print("all mailer tests passed")

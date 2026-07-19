"""Roles, invites, and role-gating tests."""
import importlib
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fresh(tmp):
    from terra import accounts
    importlib.reload(accounts)
    accounts.HOME = tmp
    accounts.DB_PATH = os.path.join(tmp, "terra.db")
    accounts.init_db()
    return accounts


def test_invite_join_roles_and_guards():
    with tempfile.TemporaryDirectory() as tmp:
        acc = _fresh(tmp)
        owner = acc.create_account("owner@t.io", "password1", "Terra")
        ws = owner["workspace_id"]
        assert owner["role"] == "owner"

        tok = acc.create_invite(ws, "mem@t.io", "member")
        joined = acc.create_account("mem@t.io", "password1", invite_token=tok)
        assert joined["workspace_id"] == ws and joined["role"] == "member"
        # invite is one-time: reusing it fails
        try:
            acc.create_account("y@t.io", "password1", invite_token=tok)
            assert False, "reused invite should fail"
        except ValueError:
            pass

        members = acc.list_members(ws)
        assert len(members) == 2
        mem = [m for m in members if m["email"] == "mem@t.io"][0]

        # role hierarchy
        owner_u = acc.session_user(owner["token"])
        mem_u = acc.session_user(joined["token"])
        assert acc.has_role(owner_u, "admin") and not acc.has_role(mem_u, "admin")
        assert acc.has_role(mem_u, "member") and acc.has_role(mem_u, "viewer")

        # promote/demote member, but never demote the last owner
        assert acc.set_role(ws, mem["id"], "admin") is True
        assert acc.set_role(ws, owner["user_id"], "member") is False   # last owner guard
        assert acc.remove_member(ws, owner["user_id"]) is False        # can't remove last owner
        assert acc.remove_member(ws, mem["id"]) is True
        assert len(acc.list_members(ws)) == 1
        print("  roles: invite -> join -> promote, last-owner guard OK")


def _call(base, path, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_role_gating_over_http():
    from http.server import ThreadingHTTPServer
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TERRA_HOME"] = tmp
        os.environ["TERRA_AUTH"] = "1"
        from terra import accounts, server
        for m in (accounts, server):
            importlib.reload(m)
        pf = server.Platform()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(pf))
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            owner = _call(base, "/api/auth/signup", "POST",
                          {"email": "own@t.io", "password": "password1", "workspace": "Terra"})[1]
            OH = {"Authorization": "Bearer " + owner["token"]}
            server.acc.set_plan(owner["user"]["workspace_id"], "fleet")

            # owner invites a viewer; the invite link carries the token
            inv = _call(base, "/api/team/invite", "POST",
                        {"email": "view@t.io", "role": "viewer"}, OH)[1]
            tok = inv["invite_link"].split("invite=")[1]
            viewer = _call(base, "/api/auth/signup", "POST",
                           {"email": "view@t.io", "password": "password1", "invite_token": tok})[1]
            VH = {"Authorization": "Bearer " + viewer["token"]}
            assert viewer["user"]["role"] == "viewer"

            # viewer can read, but cannot operate or manage
            assert _call(base, "/api/overview", "GET", None, VH)[0] == 200
            assert _call(base, "/api/control", "POST", {"autonomy": True}, VH)[0] == 403
            assert _call(base, "/api/team/invite", "POST", {"email": "z@t.io"}, VH)[0] == 403
            assert _call(base, "/api/nodes/enroll-token", "POST", {}, VH)[0] == 403
            # owner can do all of those
            assert _call(base, "/api/control", "POST", {"autonomy": False}, OH)[0] == 200
            assert _call(base, "/api/nodes/enroll-token", "POST", {}, OH)[0] == 200
            print("  roles: HTTP gating (viewer 403, owner 200), invite-join OK")
        finally:
            pf.running = False
            httpd.shutdown()


if __name__ == "__main__":
    test_invite_join_roles_and_guards()
    test_role_gating_over_http()
    print("all roles tests passed")

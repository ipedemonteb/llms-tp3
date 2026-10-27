"""Import → paquete (validate/packages.py): manual → pipreqs → variantes."""

from tprag.validate.packages import candidates, is_stdlib, normalize, pipreqs_mapping


def test_normalize():
    assert normalize("PyPDF2") == normalize("pypdf2") == "pypdf2"
    assert normalize("google_auth.oauthlib") == "google-auth-oauthlib"


def test_stdlib():
    assert is_stdlib("os.path") and is_stdlib("tomllib") and is_stdlib("unittest.mock")
    assert not is_stdlib("mock") and not is_stdlib("toml")  # se confunden con la stdlib, pero no lo son


def test_manual_wins_and_uses_longest_prefix():
    assert candidates("sklearn.linear_model") == (["scikit-learn"], "manual")
    assert candidates("google.oauth2.service_account") == (["google-auth"], "manual")
    assert candidates("google.cloud.storage") == (["google-cloud-storage"], "manual")
    # correcciones a pipreqs: `MySQLdb` → MySQL-python y `google` → protobuf allá
    assert pipreqs_mapping()["MySQLdb"] == "MySQL-python"
    assert candidates("MySQLdb") == (["mysqlclient"], "manual")


def test_pipreqs_fallback():
    assert candidates("serial.tools") == (["pyserial"], "pipreqs")


def test_variants_last():
    assert candidates("fastjsonparser") == (["fastjsonparser", "python-fastjsonparser"], "variants")
    # pipreqs tiene entradas que solo cambian mayúsculas (`requests:Requests`): inofensivas, PyPI normaliza
    assert candidates("requests") == (["Requests"], "pipreqs")


def test_pipreqs_file_is_the_pinned_copy():
    mapping = pipreqs_mapping()
    assert len(mapping) == 834 and mapping["cv2"] == "opencv-python"  # 1.157 líneas, 68 imports repetidos

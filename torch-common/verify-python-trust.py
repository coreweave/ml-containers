# SPDX-FileCopyrightText: 2026 Anlatan, Inc.
#
# SPDX-License-Identifier: MIT

"""Check OS/Python CA bundles and the active uv before publishing the image."""

import hashlib
import shutil
import ssl
import subprocess

import certifi
import pip
from pip._vendor import certifi as pip_certifi
from pip._vendor.packaging.version import Version

for module, minimum in ((pip, "26.2.2"), (certifi, "2026.10.19"), (pip_certifi, "2026.10.19")):
    if Version(module.__version__) < Version(minimum):
        raise SystemExit(f"{module.__name__} {module.__version__} is older than {minimum}")

blocked = {
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b": "emSign ECC Root CA - G3",
    "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4": "AAA Certificate Services",
}
for bundle in ("/etc/ssl/certs/ca-certificates.crt", certifi.where(), pip_certifi.where()):
    context = ssl.create_default_context(cafile=bundle)
    for certificate in context.get_ca_certs(binary_form=True):
        fingerprint = hashlib.sha256(certificate).hexdigest()
        if fingerprint in blocked:
            raise SystemExit(f"{bundle} still trusts {blocked[fingerprint]}")
    print(f"Verified CA bundle: {bundle}")

if shutil.which("uv"):
    uv_version = subprocess.check_output(["uv", "--version"], text=True).split()[1]
    if Version(uv_version) < Version("0.13.4"):
        raise SystemExit(f"uv {uv_version} is older than 0.13.4")
    print(f"Verified uv: {uv_version}")
print(f"Verified pip: {pip.__version__}; certifi: {certifi.__version__}")

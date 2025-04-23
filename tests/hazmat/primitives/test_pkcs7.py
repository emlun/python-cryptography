# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import contextlib
import datetime
import email.parser
import os
import typing
from email.message import EmailMessage

import pytest

from cryptography import exceptions, x509
from cryptography.exceptions import _Reasons
from cryptography.hazmat.bindings._rust import test_support
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.serialization import pkcs7
from tests.x509.test_x509 import _generate_ca_and_leaf

from ...hazmat.primitives.fixtures_rsa import (
    RSA_KEY_2048_ALT,
)
from ...hazmat.primitives.test_ec import (
    _skip_curve_unsupported,
    _skip_deterministic_ecdsa_unsupported,
)
from ...hazmat.primitives.test_rsa import rsa_key_2048
from ...utils import load_vectors_from_file, raises_unsupported_algorithm

# Make ruff happy since we're importing fixtures that pytest patches in as
# func args
__all__ = ["rsa_key_2048"]


@pytest.mark.supported(
    only_if=lambda backend: backend.pkcs7_supported(),
    skip_message="Requires OpenSSL with PKCS7 support",
)
class TestPKCS7Loading:
    def test_load_invalid_der_pkcs7(self, backend):
        with pytest.raises(ValueError):
            pkcs7.load_der_pkcs7_certificates(b"nonsense")

    def test_load_invalid_pem_pkcs7(self, backend):
        with pytest.raises(ValueError):
            pkcs7.load_pem_pkcs7_certificates(b"nonsense")

        with pytest.raises(ValueError):
            pkcs7.load_pem_pkcs7_certificates(b"""
-----BEGIN CERTIFICATE-----
-----END CERTIFICATE-----
            """)

    def test_not_bytes_der(self, backend):
        with pytest.raises(TypeError):
            pkcs7.load_der_pkcs7_certificates(38)  # type: ignore[arg-type]

    def test_not_bytes_pem(self, backend):
        with pytest.raises(TypeError):
            pkcs7.load_pem_pkcs7_certificates(38)  # type: ignore[arg-type]

    def test_load_pkcs7_pem(self, backend):
        certs = load_vectors_from_file(
            os.path.join("pkcs7", "isrg.pem"),
            lambda pemfile: pkcs7.load_pem_pkcs7_certificates(pemfile.read()),
            mode="rb",
        )
        assert len(certs) == 1
        assert certs[0].subject.get_attributes_for_oid(
            x509.oid.NameOID.COMMON_NAME
        ) == [x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "ISRG Root X1")]

    @pytest.mark.parametrize(
        "filepath",
        [
            os.path.join("pkcs7", "amazon-roots.der"),
            os.path.join("pkcs7", "amazon-roots.p7b"),
        ],
    )
    def test_load_pkcs7_der(self, filepath, backend):
        if filepath.endswith("p7b"):
            ctx: typing.Any = pytest.warns(UserWarning)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            certs = load_vectors_from_file(
                filepath,
                lambda derfile: pkcs7.load_der_pkcs7_certificates(
                    derfile.read()
                ),
                mode="rb",
            )
        assert len(certs) == 2
        assert certs[0].subject.get_attributes_for_oid(
            x509.oid.NameOID.COMMON_NAME
        ) == [
            x509.NameAttribute(
                x509.oid.NameOID.COMMON_NAME, "Amazon Root CA 3"
            )
        ]
        assert certs[1].subject.get_attributes_for_oid(
            x509.oid.NameOID.COMMON_NAME
        ) == [
            x509.NameAttribute(
                x509.oid.NameOID.COMMON_NAME, "Amazon Root CA 2"
            )
        ]

    def test_load_pkcs7_unsupported_type(self, backend):
        with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_SERIALIZATION):
            load_vectors_from_file(
                os.path.join("pkcs7", "enveloped.pem"),
                lambda pemfile: pkcs7.load_pem_pkcs7_certificates(
                    pemfile.read()
                ),
                mode="rb",
            )

    def test_load_pkcs7_empty_certificates(self):
        der = b"\x30\x0b\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x07\x02"

        with pytest.raises(ValueError):
            pkcs7.load_der_pkcs7_certificates(der)


def _load_cert_key():
    key = load_vectors_from_file(
        os.path.join("x509", "custom", "ca", "ca_key.pem"),
        lambda pemfile: serialization.load_pem_private_key(
            pemfile.read(), None, unsafe_skip_rsa_key_validation=True
        ),
        mode="rb",
    )
    cert = load_vectors_from_file(
        os.path.join("x509", "custom", "ca", "ca.pem"),
        loader=lambda pemfile: x509.load_pem_x509_certificate(pemfile.read()),
        mode="rb",
    )
    return cert, key


@pytest.mark.supported(
    only_if=lambda backend: backend.pkcs7_supported(),
    skip_message="Requires OpenSSL with PKCS7 support",
)
class TestPKCS7SignatureBuilder:
    def test_invalid_data(self, backend):
        builder = pkcs7.PKCS7SignatureBuilder()
        with pytest.raises(TypeError):
            builder.set_data("not bytes")  # type: ignore[arg-type]

    def test_set_data_twice(self, backend):
        builder = pkcs7.PKCS7SignatureBuilder().set_data(b"test")
        with pytest.raises(ValueError):
            builder.set_data(b"test")

    def test_sign_no_signer(self, backend):
        builder = pkcs7.PKCS7SignatureBuilder().set_data(b"test")
        with pytest.raises(ValueError):
            builder.sign(serialization.Encoding.SMIME, [])

    def test_sign_no_data(self, backend):
        cert, key = _load_cert_key()
        builder = pkcs7.PKCS7SignatureBuilder().add_signer(
            cert, key, hashes.SHA256()
        )
        with pytest.raises(ValueError):
            builder.sign(serialization.Encoding.SMIME, [])

    def test_unsupported_hash_alg(self, backend):
        cert, key = _load_cert_key()
        with pytest.raises(TypeError):
            pkcs7.PKCS7SignatureBuilder().add_signer(
                cert,
                key,
                hashes.SHA512_256(),  # type: ignore[arg-type]
            )

    def test_not_a_cert(self, backend):
        _, key = _load_cert_key()
        with pytest.raises(TypeError):
            pkcs7.PKCS7SignatureBuilder().add_signer(
                b"notacert",  # type: ignore[arg-type]
                key,
                hashes.SHA256(),
            )

    @pytest.mark.supported(
        only_if=lambda backend: backend.ed25519_supported(),
        skip_message="Does not support ed25519.",
    )
    def test_unsupported_key_type(self, backend):
        cert, _ = _load_cert_key()
        key = ed25519.Ed25519PrivateKey.generate()
        with pytest.raises(TypeError):
            pkcs7.PKCS7SignatureBuilder().add_signer(
                cert,
                key,  # type: ignore[arg-type]
                hashes.SHA256(),
            )

    def test_sign_invalid_options(self, backend):
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(b"test")
            .add_signer(cert, key, hashes.SHA256())
        )
        with pytest.raises(ValueError):
            builder.sign(
                serialization.Encoding.SMIME,
                [b"invalid"],  # type: ignore[list-item]
            )

    def test_sign_invalid_encoding(self, backend):
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(b"test")
            .add_signer(cert, key, hashes.SHA256())
        )
        with pytest.raises(ValueError):
            builder.sign(serialization.Encoding.Raw, [])

    def test_sign_invalid_options_text_no_detached(self, backend):
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(b"test")
            .add_signer(cert, key, hashes.SHA256())
        )
        options = [pkcs7.PKCS7Options.Text]
        with pytest.raises(ValueError):
            builder.sign(serialization.Encoding.SMIME, options)

    def test_sign_invalid_options_text_der_encoding(self, backend):
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(b"test")
            .add_signer(cert, key, hashes.SHA256())
        )
        options = [
            pkcs7.PKCS7Options.Text,
            pkcs7.PKCS7Options.DetachedSignature,
        ]
        with pytest.raises(ValueError):
            builder.sign(serialization.Encoding.DER, options)

    def test_sign_invalid_options_no_attrs_and_no_caps(self, backend):
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(b"test")
            .add_signer(cert, key, hashes.SHA256())
        )
        options = [
            pkcs7.PKCS7Options.NoAttributes,
            pkcs7.PKCS7Options.NoCapabilities,
        ]
        with pytest.raises(ValueError):
            builder.sign(serialization.Encoding.SMIME, options)

    def test_smime_sign_detached(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        options = [pkcs7.PKCS7Options.DetachedSignature]
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        sig = builder.sign(serialization.Encoding.SMIME, options)
        sig_binary = builder.sign(serialization.Encoding.DER, options)
        assert b"text/plain" not in sig
        # We don't have a generic ASN.1 parser available to us so we instead
        # will assert on specific byte sequences being present based on the
        # parameters chosen above.
        assert b"sha-256" in sig
        # Detached signature means that the signed data is *not* embedded into
        # the PKCS7 structure itself, but is present in the SMIME serialization
        # as a separate section before the PKCS7 data. So we should expect to
        # have data in sig but not in sig_binary
        assert data in sig
        # Parse the message to get the signed data, which is the
        # first payload in the message
        message = email.parser.BytesParser().parsebytes(sig)
        payload = message.get_payload()
        assert isinstance(payload, list)
        assert isinstance(payload[0], email.message.Message)
        signed_data = payload[0].get_payload()
        assert isinstance(signed_data, str)
        test_support.pkcs7_verify(
            serialization.Encoding.SMIME,
            sig,
            signed_data.encode(),
            [cert],
            options,
        )
        assert data not in sig_binary
        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig_binary,
            data,
            [cert],
            options,
        )

    def test_sign_byteslike(self, backend):
        data = bytearray(b"hello world")
        cert, key = _load_cert_key()
        options = [pkcs7.PKCS7Options.DetachedSignature]
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        sig = builder.sign(serialization.Encoding.SMIME, options)
        assert bytes(data) in sig
        test_support.pkcs7_verify(
            serialization.Encoding.SMIME,
            sig,
            data,
            [cert],
            options,
        )

        data = bytearray(b"")
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        sig = builder.sign(serialization.Encoding.SMIME, options)
        test_support.pkcs7_verify(
            serialization.Encoding.SMIME,
            sig,
            data,
            [cert],
            options,
        )

    @pytest.mark.parametrize(
        ("hashalg", "curve", "expect_sig_hex"),
        [
            (
                hashes.SHA256,
                ec.SECP256R1,
                "4d494d452d56657273696f6e3a20312e300d0a436f6e74656e742d547970653a206d756c7469706172742f7369676e65643b2070726f746f636f6c3d226170706c69636174696f6e2f782d706b6373372d7369676e6174757265223b206d6963616c673d227368612d323536223b20626f756e646172793d223d3d3d3d3d3d3d3d3d3d3d3d3d3d3d303331333833333431323836333638363335363d3d220d0a0d0a5468697320697320616e20532f4d494d45207369676e6564206d6573736167650d0a0d0a2d2d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d303331333833333431323836333638363335363d3d0d0a68656c6c6f20776f726c640d0a2d2d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d303331333833333431323836333638363335363d3d0d0a436f6e74656e742d547970653a206170706c69636174696f6e2f782d706b6373372d7369676e61747572653b206e616d653d22736d696d652e703773220d0a436f6e74656e742d5472616e736665722d456e636f64696e673a206261736536340d0a436f6e74656e742d446973706f736974696f6e3a206174746163686d656e743b2066696c656e616d653d22736d696d652e703773220d0a0d0a4d4949436177594a4b6f5a496876634e415163436f49494358444343416c674341514578447a414e42676c67686b67425a514d45416745464144414c42676b710d0a686b694739773042427747676767454b4d494942426a43427271414441674543416749444354414b42676771686b6a4f50515144416a414e4d517377435159440d0a5651514745774a56557a4165467730774d6a41784d4445784d6a41784d4442614677307a4d4445794d7a45774f444d774d4442614d413078437a414a42674e560d0a42415954416c56544d466b77457759484b6f5a497a6a3043415159494b6f5a497a6a304441516344516741455970305836612b597975613235586a3934615a2f0d0a5a584d4c634a6c36706d5957326d69703170656e6d327a3642417547366b424a2b424a433371716775796e3447666376414157334a73756b436838316f7075310d0a447a414b42676771686b6a4f5051514441674e484144424541694176487249646a576879546e6a512f50637a616e4f376b666b2b686741415556556f31684d490d0a6e46327745514967502f42485255672f53644757744158373166354655624278634546365558773277593948396e45364c3230786767456c4d494942495149420d0a415441544d413078437a414a42674e5642415954416c5654416749444354414e42676c67686b67425a514d4541674546414b43426f54415942676b71686b69470d0a3977304243514d784377594a4b6f5a496876634e415163424d42774743537147534962334451454a42544550467730794e5441304d6a4d784e4449794d4442610d0a4d43384743537147534962334451454a4244456942434335545365356b30302b434b557555746661666176367849547634337054674f36516950657334752f4e0d0a3654413242676b71686b694739773042435138784b54416e4d417347435743475341466c417751424b6a414c42676c67686b67425a514d45415259774377594a0d0a59495a4941575544424145434d416f4743437147534d343942414d43424567775267496841504a4e624166326c6456596e703939663654494b354537644a646c0d0a5a4f7461735a6b65585861364f4a764d4169454131757372517a584974494f704e4b6442316a765771556947325a6c353571484c77644764312b77466d4f553d0d0a0d0a2d2d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d303331333833333431323836333638363335363d3d2d2d0d0a",
            ),
        ],
    )
    def test_sign_byteslike_deterministic_ecdsa(
        self, hashalg, curve, expect_sig_hex, backend
    ):
        _skip_curve_unsupported(backend, curve())
        _skip_deterministic_ecdsa_unsupported(backend)
        if not backend.signature_hash_supported(hashalg()):
            pytest.skip(f"{hashalg} signature not supported")

        data = bytearray(b"hello world")
        cert, key = _load_cert_key()

        h = hashes.Hash(hashalg())
        h.update(b"test_build_cert_with_deterministic_ecdsa_signature.issuer")
        private_value = int.from_bytes(h.finalize(), "big")
        issuer_private_key = ec.derive_private_key(private_value, curve())
        h = hashes.Hash(hashalg())
        h.update(b"test_build_cert_with_deterministic_ecdsa_signature.subject")
        private_value = int.from_bytes(h.finalize(), "big")
        subject_private_key = ec.derive_private_key(private_value, curve())

        not_valid_before = datetime.datetime(2002, 1, 1, 12, 1)
        not_valid_after = datetime.datetime(2030, 12, 31, 8, 30)

        certbuilder = (
            x509.CertificateBuilder()
            .serial_number(777)
            .issuer_name(
                x509.Name(
                    [x509.NameAttribute(x509.oid.NameOID.COUNTRY_NAME, "US")]
                )
            )
            .subject_name(
                x509.Name(
                    [x509.NameAttribute(x509.oid.NameOID.COUNTRY_NAME, "US")]
                )
            )
            .public_key(subject_private_key.public_key())
            .not_valid_before(not_valid_before)
            .not_valid_after(not_valid_after)
        )

        cert = certbuilder.sign(
            issuer_private_key,
            hashalg(),
            backend,
            ecdsa_deterministic_signing=True,
        )

        options = [pkcs7.PKCS7Options.DetachedSignature]
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(
                cert, key, hashes.SHA256(), ecdsa_deterministic_signing=True
            )
        )

        sig = builder.sign(serialization.Encoding.SMIME, options)
        assert sig.hex() == expect_sig_hex
        assert bytes(data) in sig
        test_support.pkcs7_verify(
            serialization.Encoding.SMIME,
            sig,
            data,
            [cert],
            options,
        )

        data = bytearray(b"")
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        sig = builder.sign(serialization.Encoding.SMIME, options)
        test_support.pkcs7_verify(
            serialization.Encoding.SMIME,
            sig,
            data,
            [cert],
            options,
        )

    def test_sign_pem(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        options: typing.List[pkcs7.PKCS7Options] = []
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        sig = builder.sign(serialization.Encoding.PEM, options)
        test_support.pkcs7_verify(
            serialization.Encoding.PEM,
            sig,
            None,
            [cert],
            options,
        )

    @pytest.mark.parametrize(
        ("hash_alg", "expected_value"),
        [
            (hashes.SHA256(), b"\x06\t`\x86H\x01e\x03\x04\x02\x01"),
            (hashes.SHA384(), b"\x06\t`\x86H\x01e\x03\x04\x02\x02"),
            (hashes.SHA512(), b"\x06\t`\x86H\x01e\x03\x04\x02\x03"),
        ],
    )
    def test_sign_alternate_digests_der(
        self, hash_alg, expected_value, backend
    ):
        data = b"hello world"
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hash_alg)
        )
        options: typing.List[pkcs7.PKCS7Options] = []
        sig = builder.sign(serialization.Encoding.DER, options)
        assert expected_value in sig
        test_support.pkcs7_verify(
            serialization.Encoding.DER, sig, None, [cert], options
        )

    @pytest.mark.parametrize(
        ("hash_alg", "expected_value"),
        [
            (hashes.SHA256(), b"sha-256"),
            (hashes.SHA384(), b"sha-384"),
            (hashes.SHA512(), b"sha-512"),
        ],
    )
    def test_sign_alternate_digests_detached(
        self, hash_alg, expected_value, backend
    ):
        data = b"hello world"
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hash_alg)
        )
        options = [pkcs7.PKCS7Options.DetachedSignature]
        sig = builder.sign(serialization.Encoding.SMIME, options)
        # When in detached signature mode the hash algorithm is stored as a
        # byte string like "sha-384".
        assert expected_value in sig

    def test_sign_attached(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        options: typing.List[pkcs7.PKCS7Options] = []
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        sig_binary = builder.sign(serialization.Encoding.DER, options)
        # When not passing detached signature the signed data is embedded into
        # the PKCS7 structure itself
        assert data in sig_binary
        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig_binary,
            None,
            [cert],
            options,
        )

    def test_sign_binary(self, backend):
        data = b"hello\nworld"
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )
        options: typing.List[pkcs7.PKCS7Options] = []
        sig_no_binary = builder.sign(serialization.Encoding.DER, options)
        sig_binary = builder.sign(
            serialization.Encoding.DER, [pkcs7.PKCS7Options.Binary]
        )
        # Binary prevents translation of LF to CR+LF (SMIME canonical form)
        # so data should not be present in sig_no_binary, but should be present
        # in sig_binary
        assert data not in sig_no_binary
        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig_no_binary,
            None,
            [cert],
            options,
        )
        assert data in sig_binary
        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig_binary,
            None,
            [cert],
            options,
        )

    def test_sign_smime_canonicalization(self, backend):
        data = b"hello\nworld"
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        options: typing.List[pkcs7.PKCS7Options] = []
        sig_binary = builder.sign(serialization.Encoding.DER, options)
        # LF gets converted to CR+LF (SMIME canonical form)
        # so data should not be present in the sig
        assert data not in sig_binary
        assert b"hello\r\nworld" in sig_binary
        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig_binary,
            None,
            [cert],
            options,
        )

    def test_sign_text(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        options = [
            pkcs7.PKCS7Options.Text,
            pkcs7.PKCS7Options.DetachedSignature,
        ]
        sig_pem = builder.sign(serialization.Encoding.SMIME, options)
        # The text option adds text/plain headers to the S/MIME message
        # These headers are only relevant in SMIME mode, not binary, which is
        # just the PKCS7 structure itself.
        assert sig_pem.count(b"text/plain") == 1
        assert b"Content-Type: text/plain\r\n\r\nhello world\r\n" in sig_pem
        # Parse the message to get the signed data, which is the
        # first payload in the message
        message = email.parser.BytesParser().parsebytes(sig_pem)
        payload = message.get_payload()
        assert isinstance(payload, list)
        assert isinstance(payload[0], email.message.Message)
        signed_data = payload[0].as_bytes(
            policy=message.policy.clone(linesep="\r\n")
        )
        test_support.pkcs7_verify(
            serialization.Encoding.SMIME,
            sig_pem,
            signed_data,
            [cert],
            options,
        )

    def test_smime_capabilities(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        sig_binary = builder.sign(serialization.Encoding.DER, [])

        # 1.2.840.113549.1.9.15 (SMIMECapabilities) as an ASN.1 DER encoded OID
        assert b"\x06\t*\x86H\x86\xf7\r\x01\t\x0f" in sig_binary

        # 2.16.840.1.101.3.4.1.42 (aes256-CBC-PAD) as an ASN.1 DER encoded OID
        aes256_cbc_pad_oid = b"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x01\x2a"
        # 2.16.840.1.101.3.4.1.22 (aes192-CBC-PAD) as an ASN.1 DER encoded OID
        aes192_cbc_pad_oid = b"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x01\x16"
        # 2.16.840.1.101.3.4.1.2 (aes128-CBC-PAD) as an ASN.1 DER encoded OID
        aes128_cbc_pad_oid = b"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x01\x02"

        # Each algorithm in SMIMECapabilities should be inside its own
        # SEQUENCE.
        # This is encoded as SEQUENCE_IDENTIFIER + LENGTH + ALGORITHM_OID.
        # This tests that each algorithm is indeed encoded inside its own
        # sequence. See RFC 2633, Appendix A for more details.
        sequence_identifier = b"\x30"
        for oid in [
            aes256_cbc_pad_oid,
            aes192_cbc_pad_oid,
            aes128_cbc_pad_oid,
        ]:
            len_oid = len(oid).to_bytes(length=1, byteorder="big")
            assert sequence_identifier + len_oid + oid in sig_binary

        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig_binary,
            None,
            [cert],
            [],
        )

    def test_sign_no_capabilities(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        options = [pkcs7.PKCS7Options.NoCapabilities]
        sig_binary = builder.sign(serialization.Encoding.DER, options)
        # NoCapabilities removes the SMIMECapabilities attribute from the
        # PKCS7 structure. This is an ASN.1 sequence with the
        # OID 1.2.840.113549.1.9.15. It does NOT remove all authenticated
        # attributes, so we verify that by looking for the signingTime OID.

        # 1.2.840.113549.1.9.15 SMIMECapabilities as an ASN.1 DER encoded OID
        assert b"\x06\t*\x86H\x86\xf7\r\x01\t\x0f" not in sig_binary
        # 1.2.840.113549.1.9.5 signingTime as an ASN.1 DER encoded OID
        assert b"\x06\t*\x86H\x86\xf7\r\x01\t\x05" in sig_binary
        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig_binary,
            None,
            [cert],
            options,
        )

    def test_sign_no_attributes(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        options = [pkcs7.PKCS7Options.NoAttributes]
        sig_binary = builder.sign(serialization.Encoding.DER, options)
        # NoAttributes removes all authenticated attributes, so we shouldn't
        # find SMIMECapabilities or signingTime.

        # 1.2.840.113549.1.9.15 SMIMECapabilities as an ASN.1 DER encoded OID
        assert b"\x06\t*\x86H\x86\xf7\r\x01\t\x0f" not in sig_binary
        # 1.2.840.113549.1.9.5 signingTime as an ASN.1 DER encoded OID
        assert b"\x06\t*\x86H\x86\xf7\r\x01\t\x05" not in sig_binary
        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig_binary,
            None,
            [cert],
            options,
        )

    def test_sign_no_certs(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )

        options: typing.List[pkcs7.PKCS7Options] = []
        sig = builder.sign(serialization.Encoding.DER, options)
        assert sig.count(cert.public_bytes(serialization.Encoding.DER)) == 1

        options = [pkcs7.PKCS7Options.NoCerts]
        sig_no = builder.sign(serialization.Encoding.DER, options)
        assert sig_no.count(cert.public_bytes(serialization.Encoding.DER)) == 0

    @pytest.mark.parametrize(
        "pad",
        [
            padding.PKCS1v15(),
            None,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA512()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
        ],
    )
    def test_rsa_pkcs_padding_options(self, pad, backend):
        data = b"hello world"
        rsa_key = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_key.pem"),
            lambda pemfile: serialization.load_pem_private_key(
                pemfile.read(), None, unsafe_skip_rsa_key_validation=True
            ),
            mode="rb",
        )
        assert isinstance(rsa_key, rsa.RSAPrivateKey)
        rsa_cert = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_ca.pem"),
            loader=lambda pemfile: x509.load_pem_x509_certificate(
                pemfile.read()
            ),
            mode="rb",
        )
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(rsa_cert, rsa_key, hashes.SHA512(), rsa_padding=pad)
        )
        options: typing.List[pkcs7.PKCS7Options] = []
        sig = builder.sign(serialization.Encoding.DER, options)
        # This should be a pkcs1 sha512 signature
        if isinstance(pad, padding.PSS):
            # PKCS7_verify can't verify a PSS sig and we don't bind CMS so
            # we instead just check that a few things are present in the
            # output.
            # There should be four SHA512 OIDs in this structure
            assert sig.count(b"\x06\t`\x86H\x01e\x03\x04\x02\x03") == 4
            # There should be one MGF1 OID in this structure
            assert (
                sig.count(b"\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x08") == 1
            )
        else:
            # This should be a pkcs1 RSA signature, which uses the
            # `rsaEncryption` OID (1.2.840.113549.1.1.1) no matter which
            # digest algorithm is used.
            # See RFC 3370 section 3.2 for more details.
            # This OID appears twice, once in the certificate itself and
            # another in the SignerInfo data structure in the
            # `digest_encryption_algorithm` field.
            assert (
                sig.count(b"\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01") == 2
            )
            test_support.pkcs7_verify(
                serialization.Encoding.DER,
                sig,
                None,
                [rsa_cert],
                options,
            )

    def test_not_rsa_key_with_padding(self, backend):
        cert, key = _load_cert_key()
        with pytest.raises(TypeError):
            pkcs7.PKCS7SignatureBuilder().add_signer(
                cert, key, hashes.SHA512(), rsa_padding=padding.PKCS1v15()
            )

    def test_rsa_invalid_padding(self, backend):
        rsa_key = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_key.pem"),
            lambda pemfile: serialization.load_pem_private_key(
                pemfile.read(), None, unsafe_skip_rsa_key_validation=True
            ),
            mode="rb",
        )
        assert isinstance(rsa_key, rsa.RSAPrivateKey)
        rsa_cert = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_ca.pem"),
            loader=lambda pemfile: x509.load_pem_x509_certificate(
                pemfile.read()
            ),
            mode="rb",
        )
        with pytest.raises(TypeError):
            pkcs7.PKCS7SignatureBuilder().add_signer(
                rsa_cert,
                rsa_key,
                hashes.SHA512(),
                rsa_padding=object(),  # type: ignore[arg-type]
            )

    def test_multiple_signers(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        rsa_key = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_key.pem"),
            lambda pemfile: serialization.load_pem_private_key(
                pemfile.read(), None, unsafe_skip_rsa_key_validation=True
            ),
            mode="rb",
        )
        assert isinstance(rsa_key, rsa.RSAPrivateKey)
        rsa_cert = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_ca.pem"),
            loader=lambda pemfile: x509.load_pem_x509_certificate(
                pemfile.read()
            ),
            mode="rb",
        )
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA512())
            .add_signer(rsa_cert, rsa_key, hashes.SHA512())
        )
        options: typing.List[pkcs7.PKCS7Options] = []
        sig = builder.sign(serialization.Encoding.DER, options)
        # There should be three SHA512 OIDs in this structure
        assert sig.count(b"\x06\t`\x86H\x01e\x03\x04\x02\x03") == 3
        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig,
            None,
            [cert, rsa_cert],
            options,
        )

    def test_multiple_signers_different_hash_algs(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        rsa_key = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_key.pem"),
            lambda pemfile: serialization.load_pem_private_key(
                pemfile.read(), None, unsafe_skip_rsa_key_validation=True
            ),
            mode="rb",
        )
        rsa_cert = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_ca.pem"),
            loader=lambda pemfile: x509.load_pem_x509_certificate(
                pemfile.read()
            ),
            mode="rb",
        )
        assert isinstance(rsa_key, rsa.RSAPrivateKey)
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA384())
            .add_signer(rsa_cert, rsa_key, hashes.SHA512())
        )
        options: typing.List[pkcs7.PKCS7Options] = []
        sig = builder.sign(serialization.Encoding.DER, options)
        # There should be two SHA384 and two SHA512 OIDs in this structure
        assert sig.count(b"\x06\t`\x86H\x01e\x03\x04\x02\x02") == 2
        assert sig.count(b"\x06\t`\x86H\x01e\x03\x04\x02\x03") == 2
        test_support.pkcs7_verify(
            serialization.Encoding.DER,
            sig,
            None,
            [cert, rsa_cert],
            options,
        )

    def test_add_additional_cert_not_a_cert(self, backend):
        with pytest.raises(TypeError):
            pkcs7.PKCS7SignatureBuilder().add_certificate(
                b"notacert"  # type: ignore[arg-type]
            )

    def test_add_additional_cert(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        rsa_cert = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_ca.pem"),
            loader=lambda pemfile: x509.load_pem_x509_certificate(
                pemfile.read()
            ),
            mode="rb",
        )
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA384())
            .add_certificate(rsa_cert)
        )
        options: typing.List[pkcs7.PKCS7Options] = []
        sig = builder.sign(serialization.Encoding.DER, options)
        assert (
            sig.count(rsa_cert.public_bytes(serialization.Encoding.DER)) == 1
        )

    def test_add_multiple_additional_certs(self, backend):
        data = b"hello world"
        cert, key = _load_cert_key()
        rsa_cert = load_vectors_from_file(
            os.path.join("x509", "custom", "ca", "rsa_ca.pem"),
            loader=lambda pemfile: x509.load_pem_x509_certificate(
                pemfile.read()
            ),
            mode="rb",
        )
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA384())
            .add_certificate(rsa_cert)
            .add_certificate(rsa_cert)
        )
        options: typing.List[pkcs7.PKCS7Options] = []
        sig = builder.sign(serialization.Encoding.DER, options)
        assert (
            sig.count(rsa_cert.public_bytes(serialization.Encoding.DER)) == 2
        )


def _load_rsa_cert_key():
    key = load_vectors_from_file(
        os.path.join("x509", "custom", "ca", "rsa_key.pem"),
        lambda pemfile: serialization.load_pem_private_key(
            pemfile.read(), None, unsafe_skip_rsa_key_validation=True
        ),
        mode="rb",
    )
    cert = load_vectors_from_file(
        os.path.join("x509", "custom", "ca", "rsa_ca.pem"),
        loader=lambda pemfile: x509.load_pem_x509_certificate(pemfile.read()),
        mode="rb",
    )
    return cert, key


@pytest.mark.supported(
    only_if=lambda backend: backend.pkcs7_supported()
    and backend.rsa_encryption_supported(padding.PKCS1v15()),
    skip_message="Requires OpenSSL with PKCS7 support and PKCS1 v1.5 padding "
    "support",
)
class TestPKCS7EnvelopeBuilder:
    def test_invalid_data(self, backend):
        builder = pkcs7.PKCS7EnvelopeBuilder()
        with pytest.raises(TypeError):
            builder.set_data("not bytes")  # type: ignore[arg-type]

    def test_set_data_twice(self, backend):
        builder = pkcs7.PKCS7EnvelopeBuilder().set_data(b"test")
        with pytest.raises(ValueError):
            builder.set_data(b"test")

    def test_encrypt_no_recipient(self, backend):
        builder = pkcs7.PKCS7EnvelopeBuilder().set_data(b"test")
        with pytest.raises(ValueError):
            builder.encrypt(serialization.Encoding.SMIME, [])

    def test_encrypt_no_data(self, backend):
        cert, _ = _load_rsa_cert_key()
        builder = pkcs7.PKCS7EnvelopeBuilder().add_recipient(cert)
        with pytest.raises(ValueError):
            builder.encrypt(serialization.Encoding.SMIME, [])

    def test_unsupported_encryption(self, backend):
        cert_non_rsa, _ = _load_cert_key()
        with pytest.raises(TypeError):
            pkcs7.PKCS7EnvelopeBuilder().add_recipient(cert_non_rsa)

    def test_not_a_cert(self, backend):
        with pytest.raises(TypeError):
            pkcs7.PKCS7EnvelopeBuilder().add_recipient(
                b"notacert",  # type: ignore[arg-type]
            )

    def test_set_content_encryption_algorithm_twice(self, backend):
        builder = pkcs7.PKCS7EnvelopeBuilder()
        builder = builder.set_content_encryption_algorithm(algorithms.AES128)
        with pytest.raises(ValueError):
            builder.set_content_encryption_algorithm(algorithms.AES128)

    def test_invalid_content_encryption_algorithm(self, backend):
        class InvalidAlgorithm:
            pass

        with pytest.raises(TypeError):
            pkcs7.PKCS7EnvelopeBuilder().set_content_encryption_algorithm(
                InvalidAlgorithm,  # type: ignore[arg-type]
            )

    def test_encrypt_invalid_options(self, backend):
        cert, _ = _load_rsa_cert_key()
        builder = (
            pkcs7.PKCS7EnvelopeBuilder().set_data(b"test").add_recipient(cert)
        )
        with pytest.raises(ValueError):
            builder.encrypt(
                serialization.Encoding.SMIME,
                [b"invalid"],  # type: ignore[list-item]
            )

    def test_encrypt_invalid_encoding(self, backend):
        cert, _ = _load_rsa_cert_key()
        builder = (
            pkcs7.PKCS7EnvelopeBuilder().set_data(b"test").add_recipient(cert)
        )
        with pytest.raises(ValueError):
            builder.encrypt(serialization.Encoding.Raw, [])

    @pytest.mark.parametrize(
        "invalid_options",
        [
            [pkcs7.PKCS7Options.NoAttributes],
            [pkcs7.PKCS7Options.NoCapabilities],
            [pkcs7.PKCS7Options.NoCerts],
            [pkcs7.PKCS7Options.DetachedSignature],
            [pkcs7.PKCS7Options.Binary, pkcs7.PKCS7Options.Text],
        ],
    )
    def test_encrypt_invalid_encryption_options(
        self, backend, invalid_options
    ):
        cert, _ = _load_rsa_cert_key()
        builder = (
            pkcs7.PKCS7EnvelopeBuilder().set_data(b"test").add_recipient(cert)
        )
        with pytest.raises(ValueError):
            builder.encrypt(serialization.Encoding.DER, invalid_options)

    @pytest.mark.parametrize(
        "options",
        [
            [pkcs7.PKCS7Options.Text],
            [pkcs7.PKCS7Options.Binary],
        ],
    )
    def test_smime_encrypt_smime_encoding(self, backend, options):
        data = b"hello world\n"
        cert, private_key = _load_rsa_cert_key()
        builder = (
            pkcs7.PKCS7EnvelopeBuilder().set_data(data).add_recipient(cert)
        )
        enveloped = builder.encrypt(serialization.Encoding.SMIME, options)
        assert b"MIME-Version: 1.0\n" in enveloped
        assert b"Content-Transfer-Encoding: base64\n" in enveloped
        message = email.parser.BytesParser().parsebytes(enveloped)
        assert message.get_content_disposition() == "attachment"
        assert message.get_filename() == "smime.p7m"
        assert message.get_content_type() == "application/pkcs7-mime"
        assert message.get_param("smime-type") == "enveloped-data"
        assert message.get_param("name") == "smime.p7m"

        payload = message.get_payload(decode=True)
        assert isinstance(payload, bytes)

        # We want to know if we've serialized something that has the parameters
        # we expect, so we match on specific byte strings of OIDs & DER values.
        # OID 2.16.840.1.101.3.4.1.2 (aes128-CBC)
        assert b"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x01\x02" in payload
        # OID 1.2.840.113549.1.1.1 (rsaEncryption (PKCS #1))
        assert b"\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01" in payload
        # cryptography CA (the recipient's Common Name)
        assert (
            b"\x0c\x0f\x63\x72\x79\x70\x74\x6f\x67\x72\x61\x70\x68\x79"
            b"\x20\x43\x41"
        ) in payload

        decrypted_bytes = pkcs7.pkcs7_decrypt_smime(
            enveloped,
            cert,
            private_key,
            [o for o in options if o != pkcs7.PKCS7Options.Binary],
        )

        # New lines are canonicalized to '\r\n' when not using Binary
        expected_data = (
            data
            if pkcs7.PKCS7Options.Binary in options
            else data.replace(b"\n", b"\r\n")
        )
        assert decrypted_bytes == expected_data

    @pytest.mark.parametrize(
        "options",
        [
            [pkcs7.PKCS7Options.Text],
            [pkcs7.PKCS7Options.Binary],
        ],
    )
    def test_smime_encrypt_der_encoding(self, backend, options):
        data = b"hello world\n"
        cert, private_key = _load_rsa_cert_key()
        builder = (
            pkcs7.PKCS7EnvelopeBuilder().set_data(data).add_recipient(cert)
        )
        enveloped = builder.encrypt(serialization.Encoding.DER, options)

        # We want to know if we've serialized something that has the parameters
        # we expect, so we match on specific byte strings of OIDs & DER values.
        # OID 2.16.840.1.101.3.4.1.2 (aes128-CBC)
        assert b"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x01\x02" in enveloped
        # OID 1.2.840.113549.1.1.1 (rsaEncryption (PKCS #1))
        assert b"\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01" in enveloped
        # cryptography CA (the recipient's Common Name)
        assert (
            b"\x0c\x0f\x63\x72\x79\x70\x74\x6f\x67\x72\x61\x70\x68\x79"
            b"\x20\x43\x41"
        ) in enveloped

        decrypted_bytes = pkcs7.pkcs7_decrypt_der(
            enveloped,
            cert,
            private_key,
            [o for o in options if o != pkcs7.PKCS7Options.Binary],
        )
        # New lines are canonicalized to '\r\n' when not using Binary
        expected_data = (
            data
            if pkcs7.PKCS7Options.Binary in options
            else data.replace(b"\n", b"\r\n")
        )
        assert decrypted_bytes == expected_data

    @pytest.mark.parametrize(
        "options",
        [
            [pkcs7.PKCS7Options.Text],
            [pkcs7.PKCS7Options.Binary],
        ],
    )
    def test_smime_encrypt_pem_encoding(self, backend, options):
        data = b"hello world\n"
        cert, private_key = _load_rsa_cert_key()
        builder = (
            pkcs7.PKCS7EnvelopeBuilder().set_data(data).add_recipient(cert)
        )
        enveloped = builder.encrypt(serialization.Encoding.PEM, options)
        decrypted_bytes = pkcs7.pkcs7_decrypt_pem(
            enveloped,
            cert,
            private_key,
            [o for o in options if o != pkcs7.PKCS7Options.Binary],
        )

        # New lines are canonicalized to '\r\n' when not using Binary
        expected_data = (
            data
            if pkcs7.PKCS7Options.Binary in options
            else data.replace(b"\n", b"\r\n")
        )
        assert decrypted_bytes == expected_data

    def test_smime_encrypt_multiple_recipients(self, backend):
        data = b"hello world\n"
        cert, _ = _load_rsa_cert_key()
        builder = (
            pkcs7.PKCS7EnvelopeBuilder()
            .set_data(data)
            .add_recipient(cert)
            .add_recipient(cert)
        )
        enveloped = builder.encrypt(serialization.Encoding.DER, [])
        # cryptography CA (the recipient's Common Name)
        common_name_bytes = (
            b"\x0c\x0f\x63\x72\x79\x70\x74\x6f\x67\x72\x61"
            b"\x70\x68\x79\x20\x43\x41"
        )
        assert enveloped.count(common_name_bytes) == 2


@pytest.mark.supported(
    only_if=lambda backend: backend.pkcs7_supported()
    and backend.rsa_encryption_supported(padding.PKCS1v15()),
    skip_message="Requires OpenSSL with PKCS7 support and PKCS1 v1.5 padding "
    "support",
)
class TestPKCS7Decrypt:
    @pytest.fixture(name="data")
    def fixture_data(self, backend) -> bytes:
        return b"Hello world!\n"

    @pytest.fixture(name="certificate")
    def fixture_certificate(self, backend) -> x509.Certificate:
        certificate, _ = _load_rsa_cert_key()
        return certificate

    @pytest.fixture(name="private_key")
    def fixture_private_key(self, backend) -> rsa.RSAPrivateKey:
        _, private_key = _load_rsa_cert_key()
        return private_key

    def test_unsupported_certificate_encryption(self, backend, private_key):
        cert_non_rsa, _ = _load_cert_key()
        with pytest.raises(TypeError):
            pkcs7.pkcs7_decrypt_der(b"", cert_non_rsa, private_key, [])

    def test_not_a_cert(self, backend, private_key):
        with pytest.raises(TypeError):
            pkcs7.pkcs7_decrypt_der(b"", b"wrong_type", private_key, [])  # type: ignore[arg-type]

    def test_not_a_pkey(self, backend, certificate):
        with pytest.raises(TypeError):
            pkcs7.pkcs7_decrypt_der(b"", certificate, b"wrong_type", [])  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "invalid_options",
        [
            [b"invalid"],
            [pkcs7.PKCS7Options.NoAttributes],
            [pkcs7.PKCS7Options.Binary],
        ],
    )
    def test_pkcs7_decrypt_invalid_options(
        self, backend, invalid_options, data, certificate, private_key
    ):
        with pytest.raises(ValueError):
            pkcs7.pkcs7_decrypt_der(
                data, certificate, private_key, invalid_options
            )

    @pytest.mark.parametrize("options", [[], [pkcs7.PKCS7Options.Text]])
    def test_pkcs7_decrypt_der(
        self, backend, data, certificate, private_key, options
    ):
        # Encryption
        builder = (
            pkcs7.PKCS7EnvelopeBuilder()
            .set_data(data)
            .add_recipient(certificate)
        )
        enveloped = builder.encrypt(serialization.Encoding.DER, options)

        # Test decryption: new lines are canonicalized to '\r\n' when
        # encryption has no Binary option
        decrypted = pkcs7.pkcs7_decrypt_der(
            enveloped, certificate, private_key, options
        )
        assert decrypted == data.replace(b"\n", b"\r\n")

    def test_pkcs7_decrypt_aes_256_cbc_encrypted_content(
        self, backend, data, certificate, private_key
    ):
        # Encryption
        builder = (
            pkcs7.PKCS7EnvelopeBuilder()
            .set_data(data)
            .set_content_encryption_algorithm(algorithms.AES256)
            .add_recipient(certificate)
        )
        enveloped = builder.encrypt(serialization.Encoding.PEM, [])

        # Test decryption: new lines are canonicalized to '\r\n' when
        # encryption has no Binary option
        decrypted = pkcs7.pkcs7_decrypt_pem(
            enveloped, certificate, private_key, []
        )
        assert decrypted == data.replace(b"\n", b"\r\n")

    @pytest.mark.parametrize(
        "header",
        [
            "content-type: text/plain",
            "CONTENT-TYPE: text/plain",
            "MIME-Version: 1.0\r\nContent-Type: text/plain; charset='UTF-8'"
            "\r\nContent-Transfer-Encoding: 7bit\r\nFrom: sender@example.com"
            "\r\nTo: recipient@example.com\r\nSubject: Test Email",
        ],
    )
    def test_pkcs7_decrypt_der_text_handmade_header(
        self, backend, certificate, private_key, header
    ):
        # Encryption of data with a custom header
        base_data = "Hello world!\r\n"
        data = f"{header}\r\n\r\n{base_data}".encode()
        builder = (
            pkcs7.PKCS7EnvelopeBuilder()
            .set_data(data)
            .add_recipient(certificate)
        )
        enveloped = builder.encrypt(
            serialization.Encoding.DER, [pkcs7.PKCS7Options.Binary]
        )

        # Test decryption with text option
        decrypted = pkcs7.pkcs7_decrypt_der(
            enveloped, certificate, private_key, [pkcs7.PKCS7Options.Text]
        )
        assert decrypted == base_data.encode()

    @pytest.mark.parametrize("options", [[], [pkcs7.PKCS7Options.Text]])
    def test_pkcs7_decrypt_pem(
        self, backend, data, certificate, private_key, options
    ):
        # Encryption
        builder = (
            pkcs7.PKCS7EnvelopeBuilder()
            .set_data(data)
            .add_recipient(certificate)
        )
        enveloped = builder.encrypt(serialization.Encoding.PEM, options)

        # Test decryption: new lines are canonicalized to '\r\n' when
        # encryption has no Binary option
        decrypted = pkcs7.pkcs7_decrypt_pem(
            enveloped, certificate, private_key, options
        )
        assert decrypted == data.replace(b"\n", b"\r\n")

    def test_pkcs7_decrypt_pem_with_wrong_tag(
        self, backend, data, certificate, private_key
    ):
        with pytest.raises(ValueError):
            pkcs7.pkcs7_decrypt_pem(
                certificate.public_bytes(serialization.Encoding.PEM),
                certificate,
                private_key,
                [],
            )

    @pytest.mark.parametrize("options", [[], [pkcs7.PKCS7Options.Text]])
    def test_pkcs7_decrypt_smime(
        self, backend, data, certificate, private_key, options
    ):
        # Encryption
        builder = (
            pkcs7.PKCS7EnvelopeBuilder()
            .set_data(data)
            .add_recipient(certificate)
        )
        enveloped = builder.encrypt(serialization.Encoding.SMIME, options)

        # Test decryption
        decrypted = pkcs7.pkcs7_decrypt_smime(
            enveloped, certificate, private_key, options
        )
        assert decrypted == data.replace(b"\n", b"\r\n")

    def test_pkcs7_decrypt_no_encrypted_content(
        self, backend, data, certificate, private_key
    ):
        enveloped = load_vectors_from_file(
            os.path.join("pkcs7", "enveloped-no-content.der"),
            loader=lambda pemfile: pemfile.read(),
            mode="rb",
        )

        # Test decryption with text option
        with pytest.raises(ValueError):
            pkcs7.pkcs7_decrypt_der(enveloped, certificate, private_key, [])

    def test_pkcs7_decrypt_text_no_header(
        self, backend, data, certificate, private_key
    ):
        # Encryption of data without a header (no "Text" option)
        builder = (
            pkcs7.PKCS7EnvelopeBuilder()
            .set_data(data)
            .add_recipient(certificate)
        )
        enveloped = builder.encrypt(serialization.Encoding.DER, [])

        # Test decryption with text option
        with pytest.raises(ValueError):
            pkcs7.pkcs7_decrypt_der(
                enveloped, certificate, private_key, [pkcs7.PKCS7Options.Text]
            )

    def test_pkcs7_decrypt_text_html_content_type(
        self, backend, certificate, private_key
    ):
        # Encryption of data with a text/html content type header
        data = b"Content-Type: text/html\r\n\r\nHello world!<br>"
        builder = (
            pkcs7.PKCS7EnvelopeBuilder()
            .set_data(data)
            .add_recipient(certificate)
        )
        enveloped = builder.encrypt(
            serialization.Encoding.DER, [pkcs7.PKCS7Options.Binary]
        )

        # Test decryption with text option
        with pytest.raises(ValueError):
            pkcs7.pkcs7_decrypt_der(
                enveloped, certificate, private_key, [pkcs7.PKCS7Options.Text]
            )

    def test_smime_decrypt_no_recipient_match(
        self, backend, data, certificate, rsa_key_2048: rsa.RSAPrivateKey
    ):
        # Encrypt some data with one RSA chain
        builder = (
            pkcs7.PKCS7EnvelopeBuilder()
            .set_data(data)
            .add_recipient(certificate)
        )
        enveloped = builder.encrypt(serialization.Encoding.DER, [])

        # Prepare another RSA chain
        another_private_key = RSA_KEY_2048_ALT.private_key(
            unsafe_skip_rsa_key_validation=True
        )
        _, another_cert = _generate_ca_and_leaf(
            rsa_key_2048, another_private_key
        )

        # Test decryption with another RSA chain
        with pytest.raises(ValueError):
            pkcs7.pkcs7_decrypt_der(
                enveloped, another_cert, another_private_key, []
            )

    def test_smime_decrypt_unsupported_key_encryption_algorithm(
        self, backend, data, certificate, private_key
    ):
        enveloped = load_vectors_from_file(
            os.path.join("pkcs7", "enveloped-rsa-oaep.pem"),
            loader=lambda pemfile: pemfile.read(),
            mode="rb",
        )

        with pytest.raises(exceptions.UnsupportedAlgorithm):
            pkcs7.pkcs7_decrypt_pem(enveloped, certificate, private_key, [])

    def test_smime_decrypt_unsupported_content_encryption_algorithm(
        self, backend, data, certificate, private_key
    ):
        enveloped = load_vectors_from_file(
            os.path.join("pkcs7", "enveloped-triple-des.pem"),
            loader=lambda pemfile: pemfile.read(),
            mode="rb",
        )

        with pytest.raises(exceptions.UnsupportedAlgorithm):
            pkcs7.pkcs7_decrypt_pem(enveloped, certificate, private_key, [])

    def test_smime_decrypt_not_enveloped(
        self, backend, data, certificate, private_key
    ):
        # Create a signed email
        cert, key = _load_cert_key()
        options = [pkcs7.PKCS7Options.DetachedSignature]
        builder = (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(data)
            .add_signer(cert, key, hashes.SHA256())
        )
        signed = builder.sign(serialization.Encoding.DER, options)

        # Test decryption failure with signed email
        with pytest.raises(ValueError):
            pkcs7.pkcs7_decrypt_der(signed, certificate, private_key, [])

    def test_smime_decrypt_smime_not_encrypted(
        self, backend, certificate, private_key
    ):
        # Create a plain email
        email_message = EmailMessage()
        email_message.set_content("Hello world!")

        # Test decryption failure with plain email
        with pytest.raises(ValueError):
            pkcs7.pkcs7_decrypt_smime(
                email_message.as_bytes(), certificate, private_key, []
            )


@pytest.mark.supported(
    only_if=lambda backend: backend.pkcs7_supported(),
    skip_message="Requires OpenSSL with PKCS7 support",
)
class TestPKCS7SerializeCerts:
    @pytest.mark.parametrize(
        ("encoding", "loader"),
        [
            (serialization.Encoding.PEM, pkcs7.load_pem_pkcs7_certificates),
            (serialization.Encoding.DER, pkcs7.load_der_pkcs7_certificates),
        ],
    )
    def test_roundtrip(self, encoding, loader, backend):
        certs = load_vectors_from_file(
            os.path.join("pkcs7", "amazon-roots.der"),
            lambda derfile: pkcs7.load_der_pkcs7_certificates(derfile.read()),
            mode="rb",
        )
        p7 = pkcs7.serialize_certificates(certs, encoding)
        certs2 = loader(p7)
        assert certs == certs2

    def test_ordering(self, backend):
        certs = load_vectors_from_file(
            os.path.join("pkcs7", "amazon-roots.der"),
            lambda derfile: pkcs7.load_der_pkcs7_certificates(derfile.read()),
            mode="rb",
        )
        p7 = pkcs7.serialize_certificates(
            list(reversed(certs)), serialization.Encoding.DER
        )
        certs2 = pkcs7.load_der_pkcs7_certificates(p7)
        assert certs == certs2

    def test_pem_matches_vector(self, backend):
        p7_pem = load_vectors_from_file(
            os.path.join("pkcs7", "isrg.pem"),
            lambda p: p.read(),
            mode="rb",
        )
        certs = pkcs7.load_pem_pkcs7_certificates(p7_pem)
        p7 = pkcs7.serialize_certificates(certs, serialization.Encoding.PEM)
        assert p7 == p7_pem

    def test_der_matches_vector(self, backend):
        p7_der = load_vectors_from_file(
            os.path.join("pkcs7", "amazon-roots.der"),
            lambda p: p.read(),
            mode="rb",
        )
        certs = pkcs7.load_der_pkcs7_certificates(p7_der)
        p7 = pkcs7.serialize_certificates(certs, serialization.Encoding.DER)
        assert p7 == p7_der

    def test_invalid_types(self):
        certs = load_vectors_from_file(
            os.path.join("pkcs7", "amazon-roots.der"),
            lambda derfile: pkcs7.load_der_pkcs7_certificates(derfile.read()),
            mode="rb",
        )
        with pytest.raises(TypeError):
            pkcs7.serialize_certificates(
                object(),  # type: ignore[arg-type]
                serialization.Encoding.PEM,
            )

        with pytest.raises(TypeError):
            pkcs7.serialize_certificates([], serialization.Encoding.PEM)

        with pytest.raises(TypeError):
            pkcs7.serialize_certificates(
                certs,
                "not an encoding",  # type: ignore[arg-type]
            )


@pytest.mark.supported(
    only_if=lambda backend: not backend.pkcs7_supported(),
    skip_message="Requires OpenSSL without PKCS7 support (BoringSSL)",
)
class TestPKCS7Unsupported:
    def test_pkcs7_functions_unsupported(self):
        with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_SERIALIZATION):
            pkcs7.load_der_pkcs7_certificates(b"nonsense")

        with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_SERIALIZATION):
            pkcs7.load_pem_pkcs7_certificates(b"nonsense")


@pytest.mark.supported(
    only_if=lambda backend: backend.pkcs7_supported()
    and not backend.rsa_encryption_supported(padding.PKCS1v15()),
    skip_message="Requires OpenSSL with no PKCS1 v1.5 padding support",
)
class TestPKCS7EnvelopeBuilderUnsupported:
    def test_envelope_builder_unsupported(self, backend):
        with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_PADDING):
            pkcs7.PKCS7EnvelopeBuilder()


@pytest.mark.supported(
    only_if=lambda backend: backend.pkcs7_supported()
    and not backend.rsa_encryption_supported(padding.PKCS1v15()),
    skip_message="Requires OpenSSL with no PKCS1 v1.5 padding support",
)
class TestPKCS7DecryptUnsupported:
    def test_pkcs7_decrypt_unsupported(self, backend):
        cert, key = _load_rsa_cert_key()
        with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_PADDING):
            pkcs7.pkcs7_decrypt_der(b"", cert, key, [])

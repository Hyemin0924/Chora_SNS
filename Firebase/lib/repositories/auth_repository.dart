import 'dart:typed_data';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:mime/mime.dart';

class AuthRepository {
  final FirebaseAuth firebaseAuth;
  final FirebaseStorage firebaseStorage;
  final FirebaseFirestore firebaseFirestore;

  const AuthRepository({
    required this.firebaseAuth,
    required this.firebaseStorage,
    required this.firebaseFirestore,
  });

  Future<void> signOut() async {
    await firebaseAuth.signOut();
  }

  Future<void> signIn({
    required String email,
    required String password,
  }) async {
    try {
      UserCredential userCredential =
          await firebaseAuth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      bool isVerified = userCredential.user!.emailVerified;
      if (!isVerified) {
        await userCredential.user!.sendEmailVerification();
        await firebaseAuth.signOut();
        throw CustomException(
          code: 'Exception',
          message: '인증되지 않은 이메일',
        );
      }
    } on FirebaseException catch (e) {
      throw CustomException(
        code: e.code,
        message: e.message!,
      );
    } catch (e) {
      throw CustomException(
        code: 'Exception',
        message: e.toString(),
      );
    }
  }

  Future<void> signUp({
    required String email,
    required String name,
    required String password,
    required Uint8List? profileImage,
  }) async {
    try {
      UserCredential userCredential =
          await firebaseAuth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );

      String uid = userCredential.user!.uid;

      await userCredential.user!.sendEmailVerification();

      String? downloadURL = null;

      if (profileImage != null) {
        // mime 패키지의 lookupMimeType 함수를 사용해서 file 의 mimeType 을 String 으로 받음
        // lookupMimeType 의 첫 번째 인자값은 파일의 경로를 전달, 선택 인자값 headerBytes 에 파일의 데이터를 int 로 갖고 있는 List 를 전달
        // 원래 lookupMimeType 함수는 파일의 경로에 존재하는 파일의 확장자로부터 mimeType 을 특정하지만, headerBytes 에 파일 데이터가 전달되면
        // 파일 데이터에서 magic-number(파일의 유형에 대한 정보를 갖고 있는 데이터)로 mimeType 을 특정함
        String? mimeType = lookupMimeType('', headerBytes: profileImage);
        SettableMetadata metadata = SettableMetadata(contentType: mimeType);

        Reference ref = firebaseStorage.ref().child('profile').child(uid);
        TaskSnapshot snapshot = await ref.putData(profileImage, metadata);
        downloadURL = await snapshot.ref.getDownloadURL();
      }

      await firebaseFirestore.collection('users').doc(uid).set({
        'uid': uid,
        'email': email,
        'name': name,
        'profileImage': downloadURL,
        'feedCount': 0,
        'likes': [],
        'followers': [],
        'following': [],
      });

      firebaseAuth.signOut();
    } on FirebaseException catch (e) {
      throw CustomException(
        code: e.code,
        message: e.message!,
      );
    } catch (e) {
      throw CustomException(
        code: 'Exception',
        message: e.toString(),
      );
    }
  }
}

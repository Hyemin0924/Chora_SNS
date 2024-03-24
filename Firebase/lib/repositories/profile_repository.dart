import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/user_model.dart';

class ProfileRepository {
  final FirebaseFirestore firebaseFirestore;

  const ProfileRepository({
    required this.firebaseFirestore,
  });

  Future<UserModel> followUser({
    required String currentUserId,
    required String followId,
  }) async {
    try {
      DocumentReference<Map<String, dynamic>> currentUserDocRef =
      firebaseFirestore.collection('users').doc(currentUserId);
      DocumentReference<Map<String, dynamic>> followUserDocRef =
      firebaseFirestore.collection('users').doc(followId);

      // DocumentSnapshot<Map<String, dynamic>> currentUserSnapshot = await currentUserDocRef.get();
      // List<String> following = List<String>.from(currentUserSnapshot.data()!['following']);

      List<String> following = await currentUserDocRef
          .get()
          .then((value) => List<String>.from(value.data()!['following']));

      WriteBatch batch = firebaseFirestore.batch();

      if (following.contains(followId)) {
        batch.update(currentUserDocRef, {
          'following': FieldValue.arrayRemove([followId])
        });
        batch.update(followUserDocRef, {
          'followers': FieldValue.arrayRemove([currentUserId])
        });
      } else {
        batch.update(currentUserDocRef, {
          'following': FieldValue.arrayUnion([followId])
        });
        batch.update(followUserDocRef, {
          'followers': FieldValue.arrayUnion([currentUserId])
        });
      }

      batch.commit();

      Map<String, dynamic> map = await followUserDocRef.get().then((value) => value.data()!);
      return UserModel.fromMap(map);
    } catch (e) {
      throw CustomException(
        code: 'Exception',
        message: e.toString(),
      );
    }
  }

  Future<UserModel> getProfile({
    required String uid,
  }) async {
    try {
      DocumentSnapshot<Map<String, dynamic>> snapshot =
          await firebaseFirestore.collection('users').doc(uid).get();
      return UserModel.fromMap(snapshot.data()!);
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

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/comment_model.dart';
import 'package:instagram_clone/models/user_model.dart';
import 'package:uuid/uuid.dart';

class CommentRepository {
  final FirebaseFirestore firebaseFirestore;

  const CommentRepository({
    required this.firebaseFirestore,
  });

  Future<List<CommentModel>> getCommentList({
    required String feedId,
  }) async {
    try {
      QuerySnapshot<Map<String, dynamic>> snapshot = await firebaseFirestore
          .collection('feeds')
          .doc(feedId)
          .collection('comments')
          .orderBy('createdAt', descending: true)
          .get();

      List<CommentModel> commentList =
      await Future.wait(snapshot.docs.map((e) async {
        Map<String, dynamic> data = e.data();
        DocumentReference<Map<String, dynamic>> writerDocRef = data['writer'];
        Map<String, dynamic> writerMapData =
        await writerDocRef.get().then((value) => value.data()!);
        data['writer'] = UserModel.fromMap(writerMapData);
        return CommentModel.fromMap(data);
      }).toList());
      return commentList;
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

  Future<CommentModel> uploadComment({
    required String feedId,
    required String uid,
    required String comment,
  }) async {
    try {
      String commentId = Uuid().v1();

      DocumentReference<Map<String, dynamic>> writerDocRef =
      firebaseFirestore.collection('users').doc(uid);
      DocumentReference<Map<String, dynamic>> feedDocRef =
      firebaseFirestore.collection('feeds').doc(feedId);
      DocumentReference<Map<String, dynamic>> commentDocRef =
      feedDocRef.collection('comments').doc(commentId);

      await firebaseFirestore.runTransaction((transaction) async {
        transaction.set(commentDocRef, {
          'commentId': commentId,
          'comment': comment,
          'writer': writerDocRef,
          'createdAt': Timestamp.now(),
        });

        transaction.update(feedDocRef, {
          'commentCount': FieldValue.increment(1),
        });
      });

      UserModel userModel = await writerDocRef.get().then((snapshot) =>
      snapshot.data()!).then((data) => UserModel.fromMap(data));

      CommentModel commentModel = await commentDocRef.get().then((snapshot) => snapshot.data()!).then((data) {
        data['writer'] = userModel;
        return CommentModel.fromMap(data);
      });
      return commentModel;
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

import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/feed_model.dart';
import 'package:instagram_clone/models/user_model.dart';
import 'package:uuid/uuid.dart';

class FeedRepository {
  final FirebaseStorage firebaseStorage;
  final FirebaseFirestore firebaseFirestore;

  const FeedRepository({
    required this.firebaseStorage,
    required this.firebaseFirestore,
  });

  Future<void> deleteFeed({
    required FeedModel feedModel,
  }) async {
    try {
      WriteBatch batch = firebaseFirestore.batch();
      DocumentReference<Map<String, dynamic>> feedDocRef =
          firebaseFirestore.collection('feeds').doc(feedModel.feedId);
      DocumentReference<Map<String, dynamic>> writerDocRef =
          firebaseFirestore.collection('users').doc(feedModel.uid);

      // 해당 게시물에 좋아요를 누른 users 문서의 likes 필드에서 feedId 삭제
      List<String> likes = await feedDocRef
          .get()
          .then((value) => List<String>.from(value.data()!['likes']));

      likes.forEach((uid) {
        batch.update(firebaseFirestore.collection('users').doc(uid), {
          'likes': FieldValue.arrayRemove([feedModel.feedId]),
        });
      });

      // 해당 게시물의 comments 컬렉션의 docs 를 삭제
      QuerySnapshot<Map<String, dynamic>> commentQuerySnapshot =
          await feedDocRef.collection('comments').get();
      for (var doc in commentQuerySnapshot.docs) {
        batch.delete(doc.reference);
      }

      // feeds 컬렉션에서 문서 삭제
      batch.delete(feedDocRef);

      // 게시물 작성자의 users 문서에서 feedCount 1 감소
      batch.update(writerDocRef, {
        'feedCount': FieldValue.increment(-1),
      });

      // storage 의 이미지 삭제
      feedModel.imageUrls.forEach((element) async {
        await firebaseStorage.refFromURL(element).delete();
      });

      batch.commit();
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

  Future<FeedModel> likeFeed({
    required String feedId,
    required List<String> feedLikes,
    required String uid,
    required List<String> userLikes,
  }) async {
    try {
      DocumentReference<Map<String, dynamic>> userDocRef =
          firebaseFirestore.collection('users').doc(uid);
      DocumentReference<Map<String, dynamic>> feedDocRef =
          firebaseFirestore.collection('feeds').doc(feedId);

      // 게시물을 좋아하는 유저 목록에 uid 가 포함되어 있는지 확인
      // 포함되어 있다면 좋아요 취소
      // 게시물의 likes 필드에서 uid 삭제
      // 게시물의 likeCount 를 1 감소

      // 유저가 좋아하는 게시물 목록에 feedId 가 포함되어 있는지 확인
      // 포함되어 있다면 좋아요 취소
      // 유저의 likes 필드에서 feedId 삭제
      await firebaseFirestore.runTransaction((transaction) async {
        bool isFeedContains = feedLikes.contains(uid);

        transaction.update(feedDocRef, {
          'likes': isFeedContains
              ? FieldValue.arrayRemove([uid])
              : FieldValue.arrayUnion([uid]),
          'likeCount': isFeedContains
              ? FieldValue.increment(-1)
              : FieldValue.increment(1),
        });

        transaction.update(userDocRef, {
          'likes': userLikes.contains(feedId)
              ? FieldValue.arrayRemove([feedId])
              : FieldValue.arrayUnion([feedId]),
        });
      });

      Map<String, dynamic> feedMapData =
          await feedDocRef.get().then((value) => value.data()!);

      DocumentReference<Map<String, dynamic>> writerDocRef =
          feedMapData['writer'];
      Map<String, dynamic> userMapData =
          await writerDocRef.get().then((value) => value.data()!);
      UserModel userModel = UserModel.fromMap(userMapData);
      feedMapData['writer'] = userModel;
      return FeedModel.fromMap(feedMapData);
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

  Future<List<FeedModel>> getFeedList({
    String? uid,
    String? feedId,
    required int feedLength,
  }) async {
    try {
      // 전체 피드 검색
      Query<Map<String, dynamic>> query = await firebaseFirestore
          .collection('feeds')
          //.where('uid',  isEqualTo: uid) // firebase 업데이트로 인해 사용 불가
          .orderBy('createAt', descending: true)
          .limit(feedLength);

      // uid 가 null 이 아닐 경우(특정 유저의 피드를 가져올 경우) 조건 추가
      if (uid != null) {
        query = query.where('uid', isEqualTo: uid);
      }

      if (feedId != null) {
        DocumentSnapshot<Map<String, dynamic>> startDocRef =
            await firebaseFirestore.collection('feeds').doc(feedId).get();
        query = query.startAfterDocument(startDocRef);
      }

      QuerySnapshot<Map<String, dynamic>> snapshot = await query.get();
      return await Future.wait(snapshot.docs.map((e) async {
        Map<String, dynamic> data = e.data();
        DocumentReference<Map<String, dynamic>> writerDocRef = data['writer'];
        DocumentSnapshot<Map<String, dynamic>> writerSnapshot =
            await writerDocRef.get();
        UserModel userModel = UserModel.fromMap(writerSnapshot.data()!);
        data['writer'] = userModel;
        return FeedModel.fromMap(data);
      }).toList());
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

  Future<FeedModel> uploadFeed({
    required List<String> files,
    required String desc,
    required String uid,
  }) async {
    List<String> imageUrls = [];

    try {
      WriteBatch batch = firebaseFirestore.batch();

      String feedId = Uuid().v1();

      // firestore 문서 참조
      DocumentReference<Map<String, dynamic>> feedDocRef =
          firebaseFirestore.collection('feeds').doc(feedId);

      DocumentReference<Map<String, dynamic>> userDocRef =
          firebaseFirestore.collection('users').doc(uid);

      // storage 참조
      Reference ref = firebaseStorage.ref().child('feeds').child(feedId);

      imageUrls = await Future.wait(files.map((e) async {
        String imageId = Uuid().v1();
        TaskSnapshot taskSnapshot = await ref.child(imageId).putFile(File(e));
        return await taskSnapshot.ref.getDownloadURL();
      }).toList());

      DocumentSnapshot<Map<String, dynamic>> userSnapshot =
          await userDocRef.get();
      UserModel userModel = UserModel.fromMap(userSnapshot.data()!);

      FeedModel feedModel = FeedModel.fromMap({
        'uid': uid,
        'feedId': feedId,
        'desc': desc,
        'imageUrls': imageUrls,
        'likes': [],
        'likeCount': 0,
        'commentCount': 0,
        'createAt': Timestamp.now(),
        'writer': userModel,
      });

      batch.set(feedDocRef, feedModel.toMap(userDocRef: userDocRef));

      batch.update(userDocRef, {
        'feedCount': FieldValue.increment(1),
      });

      batch.commit();
      return feedModel;
    } on FirebaseException catch (e) {
      _deleteImage(imageUrls);
      throw CustomException(
        code: e.code,
        message: e.message!,
      );
    } catch (e) {
      _deleteImage(imageUrls);
      throw CustomException(
        code: 'Exception',
        message: e.toString(),
      );
    }
  }

  void _deleteImage(List<String> imageUrls) {
    imageUrls.forEach((element) async {
      await firebaseStorage.refFromURL(element).delete();
    });
  }
}

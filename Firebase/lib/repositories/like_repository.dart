import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/feed_model.dart';
import 'package:instagram_clone/models/user_model.dart';

class LikeRepository {
  final FirebaseFirestore firebaseFirestore;

  const LikeRepository({
    required this.firebaseFirestore,
  });

  Future<List<FeedModel>> getLikeList({
    required String uid,
    required int likeLength,
    String? feedId,
  }) async {
    try {
      Map<String, dynamic> userMapData = await firebaseFirestore
          .collection('users')
          .doc(uid)
          .get()
          .then((value) => value.data()!);

      List<String> likes = List<String>.from(userMapData['likes']);

      if (feedId != null) {
        int startIdx = likes.indexWhere((element) => element == feedId) + 1;
        int endIdx = (startIdx + likeLength) > likes.length
            ? likes.length
            : startIdx + likeLength;
        likes = likes.sublist(startIdx, endIdx);
      } else {
        int endIdx = likes.length < likeLength ? likes.length : likeLength;
        likes = likes.sublist(0, endIdx);
      }

      List<FeedModel> likeList = await Future.wait(likes.map((feedId) async {
        DocumentSnapshot<Map<String, dynamic>> documentSnapshot =
            await firebaseFirestore.collection('feeds').doc(feedId).get();
        Map<String, dynamic> feedMapData = documentSnapshot.data()!;
        DocumentReference<Map<String, dynamic>> userDocRef =
            feedMapData['writer'];
        Map<String, dynamic> writerMapData =
            await userDocRef.get().then((value) => value.data()!);
        feedMapData['writer'] = UserModel.fromMap(writerMapData);
        return FeedModel.fromMap(feedMapData);
      }).toList());
      return likeList;
    } catch (e) {
      throw CustomException(
        code: 'Exception',
        message: e.toString(),
      );
    }
  }
}

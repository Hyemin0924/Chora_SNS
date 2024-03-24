import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:instagram_clone/models/user_model.dart';

class FeedModel {
  final String uid;
  final String feedId;
  final String desc;
  final List<String> imageUrls;
  final List<String> likes;
  final int commentCount;
  final int likeCount;
  final Timestamp createAt;
  final UserModel writer;

  const FeedModel({
    required this.uid,
    required this.feedId,
    required this.desc,
    required this.imageUrls,
    required this.likes,
    required this.commentCount,
    required this.likeCount,
    required this.createAt,
    required this.writer,
  });

  Map<String, dynamic> toMap({
    required DocumentReference<Map<String, dynamic>> userDocRef,
  }) {
    return {
      'uid': this.uid,
      'feedId': this.feedId,
      'desc': this.desc,
      'imageUrls': this.imageUrls,
      'likes': this.likes,
      'commentCount': this.commentCount,
      'likeCount': this.likeCount,
      'createAt': this.createAt,
      'writer': userDocRef,
    };
  }

  factory FeedModel.fromMap(Map<String, dynamic> map) {
    return FeedModel(
      uid: map['uid'],
      feedId: map['feedId'],
      desc: map['desc'],
      imageUrls: List<String>.from(map['imageUrls']),
      likes: List<String>.from(map['likes']),
      commentCount: map['commentCount'],
      likeCount: map['likeCount'],
      createAt: map['createAt'],
      writer: map['writer'],
    );
  }

  @override
  String toString() {
    return 'FeedModel{uid: $uid, feedId: $feedId, desc: $desc, imageUrls: $imageUrls, likes: $likes, commentCount: $commentCount, likeCount: $likeCount, createAt: $createAt, writer: $writer}';
  }
}

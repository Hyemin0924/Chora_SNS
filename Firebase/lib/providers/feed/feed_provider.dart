import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_state_notifier/flutter_state_notifier.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/feed_model.dart';
import 'package:instagram_clone/models/user_model.dart';
import 'package:instagram_clone/providers/feed/feed_state.dart';
import 'package:instagram_clone/providers/user/user_state.dart';
import 'package:instagram_clone/repositories/feed_repository.dart';

class FeedProvider extends StateNotifier<FeedState> with LocatorMixin {
  FeedProvider() : super(FeedState.init());

  Future<void> deleteFeed({
    required FeedModel feedModel,
  }) async {
    state = state.copyWith(feedStatus: FeedStatus.submitting);

    try {
      await read<FeedRepository>().deleteFeed(feedModel: feedModel);

      List<FeedModel> newFeedList = state.feedList
          .where((element) => element.feedId != feedModel.feedId)
          .toList();

      state = state.copyWith(
        feedStatus: FeedStatus.success,
        feedList: newFeedList,
      );
    } on CustomException catch (_) {
      state = state.copyWith(feedStatus: FeedStatus.error);
      rethrow;
    }
  }

  Future<FeedModel> likeFeed({
    required String feedId,
    required List<String> feedLikes,
  }) async {
    state = state.copyWith(feedStatus: FeedStatus.submitting);

    try {
      UserModel userModel = read<UserState>().userModel;

      FeedModel feedModel = await read<FeedRepository>().likeFeed(
        feedId: feedId,
        feedLikes: feedLikes,
        uid: userModel.uid,
        userLikes: userModel.likes,
      );

      List<FeedModel> newFeedList = state.feedList.map((feed) {
        return feed.feedId == feedId ? feedModel : feed;
      }).toList();

      state = state.copyWith(
        feedStatus: FeedStatus.success,
        feedList: newFeedList,
      );

      return feedModel;
    } on CustomException catch (_) {
      state = state.copyWith(feedStatus: FeedStatus.error);
      rethrow;
    }
  }

  Future<void> getFeedList({
    String? feedId,
  }) async {
    final int feedLength = 3;

    try {
      state = feedId == null
          ? state.copyWith(feedStatus: FeedStatus.fetching)
          : state.copyWith(feedStatus: FeedStatus.reFetching);

      List<FeedModel> feedList = await read<FeedRepository>().getFeedList(
        feedLength: feedLength,
        feedId: feedId,
      );

      List<FeedModel> newFeedList = [
        if (feedId != null) ...state.feedList,
        ...feedList,
      ];

      state = state.copyWith(
        feedList: newFeedList,
        feedStatus: FeedStatus.success,
        hasNext: feedList.length == feedLength,
      );
    } on CustomException catch (_) {
      state = state.copyWith(feedStatus: FeedStatus.error);
      rethrow;
    }
  }

  Future<void> uploadFeed({
    required List<String> files,
    required String desc,
  }) async {
    try {
      state = state.copyWith(feedStatus: FeedStatus.submitting);

      String uid = read<User>().uid;
      FeedModel feedModel = await read<FeedRepository>().uploadFeed(
        files: files,
        desc: desc,
        uid: uid,
      );

      state = state.copyWith(
        feedStatus: FeedStatus.success,
        feedList: [feedModel, ...state.feedList],
      );
    } on CustomException catch (_) {
      state = state.copyWith(feedStatus: FeedStatus.error);
      rethrow;
    }
  }
}

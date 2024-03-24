import 'package:flutter_state_notifier/flutter_state_notifier.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/feed_model.dart';
import 'package:instagram_clone/providers/like/like_state.dart';
import 'package:instagram_clone/providers/user/user_state.dart';
import 'package:instagram_clone/repositories/like_repository.dart';

class LikeProvider extends StateNotifier<LikeState> with LocatorMixin {
  LikeProvider() : super(LikeState.init());

  void deleteFeed({
    required String feedId,
  }) {
    state = state.copyWith(likeStatus: LikeStatus.submitting);

    try {
      List<FeedModel> newLikeList =
          state.likeList.where((element) => element.feedId != feedId).toList();

      state = state.copyWith(
        likeStatus: LikeStatus.success,
        likeList: newLikeList,
      );
    } on CustomException catch (_) {
      state = state.copyWith(likeStatus: LikeStatus.error);
      rethrow;
    }
  }

  void likeFeed({
    required FeedModel newFeedModel,
  }) {
    state = state.copyWith(likeStatus: LikeStatus.submitting);

    try {
      List<FeedModel> newLikeList = [];

      int index = state.likeList
          .indexWhere((feedModel) => feedModel.feedId == newFeedModel.feedId);

      if (index == -1) {
        newLikeList = [newFeedModel, ...state.likeList];
      } else {
        state.likeList.removeAt(index);
        newLikeList = state.likeList.toList();
      }

      state = state.copyWith(
        likeStatus: LikeStatus.success,
        likeList: newLikeList,
      );
    } on CustomException catch (_) {
      state = state.copyWith(likeStatus: LikeStatus.error);
      rethrow;
    }
  }

  Future<void> getLikeList({
    String? feedId,
  }) async {
    final int likeLength = 3;
    state = feedId == null
        ? state.copyWith(likeStatus: LikeStatus.fetching)
        : state.copyWith(likeStatus: LikeStatus.reFetching);

    try {
      String uid = read<UserState>().userModel.uid;
      List<FeedModel> likeList = await read<LikeRepository>().getLikeList(
        uid: uid,
        feedId: feedId,
        likeLength: likeLength,
      );

      List<FeedModel> newLikeList = [
        if (feedId != null) ...state.likeList,
        ...likeList,
      ];

      state = state.copyWith(
        likeStatus: LikeStatus.success,
        likeList: newLikeList,
        hasNext: likeList.length == likeLength,
      );
    } on CustomException catch (_) {
      state = state.copyWith(likeStatus: LikeStatus.error);
      rethrow;
    }
  }
}

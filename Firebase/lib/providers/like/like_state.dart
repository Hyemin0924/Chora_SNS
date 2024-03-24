import 'package:instagram_clone/models/feed_model.dart';

enum LikeStatus {
  init,
  submitting,
  fetching,
  reFetching,
  success,
  error,
}

class LikeState {
  final LikeStatus likeStatus;
  final List<FeedModel> likeList;
  final bool hasNext;

  const LikeState({
    required this.likeStatus,
    required this.likeList,
    required this.hasNext,
  });

  factory LikeState.init() {
    return LikeState(
      likeStatus: LikeStatus.init,
      likeList: [],
      hasNext: true,
    );
  }

  LikeState copyWith({
    LikeStatus? likeStatus,
    List<FeedModel>? likeList,
    bool? hasNext,
  }) {
    return LikeState(
      likeStatus: likeStatus ?? this.likeStatus,
      likeList: likeList ?? this.likeList,
      hasNext: hasNext ?? this.hasNext,
    );
  }

  @override
  String toString() {
    return 'LikeState{likeStatus: $likeStatus, likeList: $likeList, hasNext: $hasNext}';
  }
}

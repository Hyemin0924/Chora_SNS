import 'package:instagram_clone/models/feed_model.dart';

enum FeedStatus {
  init,
  submitting,
  fetching,
  reFetching,
  success,
  error,
}

class FeedState {
  final FeedStatus feedStatus;
  final List<FeedModel> feedList;
  final bool hasNext;

  const FeedState({
    required this.feedStatus,
    required this.feedList,
    required this.hasNext,
  });

  factory FeedState.init() {
    return FeedState(
      feedStatus: FeedStatus.init,
      feedList: [],
      hasNext: true,
    );
  }

  FeedState copyWith({
    FeedStatus? feedStatus,
    List<FeedModel>? feedList,
    bool? hasNext,
  }) {
    return FeedState(
      feedStatus: feedStatus ?? this.feedStatus,
      feedList: feedList ?? this.feedList,
      hasNext: hasNext ?? this.hasNext,
    );
  }
}

import 'package:instagram_clone/models/feed_model.dart';
import 'package:instagram_clone/models/user_model.dart';

enum ProfileStatus {
  init,
  submitting,
  fetching,
  reFetching,
  success,
  error,
}

class ProfileState {
  final ProfileStatus profileStatus;
  final UserModel userModel;
  final List<FeedModel> feedList;
  final bool hasNext;

  const ProfileState({
    required this.profileStatus,
    required this.userModel,
    required this.feedList,
    required this.hasNext,
  });

  factory ProfileState.init() {
    return ProfileState(
      profileStatus: ProfileStatus.init,
      userModel: UserModel.init(),
      feedList: [],
      hasNext: true,
    );
  }

  ProfileState copyWith({
    ProfileStatus? profileStatus,
    UserModel? userModel,
    List<FeedModel>? feedList,
    bool? hasNext,
  }) {
    return ProfileState(
      profileStatus: profileStatus ?? this.profileStatus,
      userModel: userModel ?? this.userModel,
      feedList: feedList ?? this.feedList,
      hasNext: hasNext ?? this.hasNext,
    );
  }
}

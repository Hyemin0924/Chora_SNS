import 'package:instagram_clone/models/user_model.dart';

enum SearchStatus {
  init,
  searching,
  success,
  error,
}

class SearchState {
  final SearchStatus searchStatus;
  final List<UserModel> userModelList;

  const SearchState({
    required this.searchStatus,
    required this.userModelList,
  });

  factory SearchState.init() {
    return SearchState(
      searchStatus: SearchStatus.init,
      userModelList: [],
    );
  }

  SearchState copyWith({
    SearchStatus? searchStatus,
    List<UserModel>? userModelList,
  }) {
    return SearchState(
      searchStatus: searchStatus ?? this.searchStatus,
      userModelList: userModelList ?? this.userModelList,
    );
  }

  @override
  String toString() {
    return 'SearchState{searchStatus: $searchStatus, userModelList: $userModelList}';
  }
}

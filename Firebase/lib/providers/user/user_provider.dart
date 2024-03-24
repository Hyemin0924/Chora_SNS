import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_state_notifier/flutter_state_notifier.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/user_model.dart';
import 'package:instagram_clone/providers/user/user_state.dart';
import 'package:instagram_clone/repositories/profile_repository.dart';

class UserProvider extends StateNotifier<UserState> with LocatorMixin {
  UserProvider() : super(UserState.init());

  Future<void> getUserInfo() async {
    try {
      String uid = read<User>().uid;
      UserModel userModel = await read<ProfileRepository>().getProfile(uid: uid);
      state = state.copyWith(userModel: userModel);
    } on CustomException catch (_) {
      rethrow;
    }
  }
}
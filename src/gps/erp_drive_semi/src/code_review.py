1. optimal throttle.cpp  
헤더에는 typedef enum DRIVE/ STOP_LINE/ STOP_DELIVERY/ DECEL/
                    DECEL_SPEED_BUMP_START/ DECEL_SPEED_BUMP_END 
                    
        namespace TrafficLightStatus 에 const로 신호등 관련 정리되어있음
        EventPlaceInfo 구조체에는 x,y 좌표와 type이 들어감

        EventPlaceInfoManager 클래스
        {
            <private>
            getCurrEventPlaceInfo 를 통해 현재 상태 반환
            nextEventPlaceInfo 를 통해 true false를 반환 (미션이 다 끝났건지 체크 > iter가 더 순회할 부분이 x)
            calcCurrPoseToEventPlaceDist 를 통해 curr_pose 와 getCurrEventPlaceInfo를 통해 미션과의 거리를 비교함

            <public>
            event_place 라는 벡터안에 EventPlaceInfo 타입 객체들을 인자로 갖는다.
            
        }
        
        OptimalThrottle 클래스

            const double DISTANCE = 20.0; >>> 이벤트가 발생하기 20m전부터 감지하도록
            const double STOP_TIME = 3.0; >>> 주차 시 정지 시간 
            const double MAX_VELOCITY = 20.0; 
            const double DECEL_VELOCITY = 5.0;

            bool shouldStop, shouldDecel;
            double throttle;

        calcLinearVelocity >> 속도 관련함수
            if(!shouldStop) return MAX_VELOCITY >>>shouldStop 플래그는 언제 바뀜?
            shouldStop이 True일 경우 calcCurrPoseToEventPlaceDist 를 통해 거리를 구하고 함수값에 맞게 감속을 하도록 속도값 리턴함
        calcTanhVelocity (위와 마찬가지 로직, 함수식만 다르다) 

2. controller.cpp
헤더에는 optimal_throttle include 
        클래스 Controller(char control_target) >>> control_target 에는 e또는 morai_drive_pub       

        // 속도 | rosparam 으로 관리될 값
        double velocity;
        OptimalThrottle optimal_throttle;

        // 조향각
        double steering;
        double brake;



        <<<<<<<<<<<<<<<<< 퍼블리셔 >>>>>>>>>>>>>>>>>>>
        1. 제어 메시지 (erp일 경우 control value morai일 경우 /ctrl_cmd)
        2. curr_idx (planner에서 local path를 생성하는 데 사용하기 위함)
        <<<<<<<<<<<<<<<<< 콜백함수 >>>>>>>>>>>>>>>>>>>
        1. path 를 받음 (global_path, local_path, parking_path)
        2. current_pose 를 받음 
            <<<<<<<< optimal_throttle.eventPlaceInfoManager 사용 >>>>>>>>
            # if. 정지 혹은 감속이면 **shouldStop 은 브레이킹 하는거고 shouldDecel 은 말그대로 감속임**
                / if. 500mm이상 남은 경우는 그냥 return 
                / elif. 정지선이면 
                        (10초 정지하고 지난경우는 shouldDecel, shouldStop 둘다 false하고 nextEvent로 넘어감)
                / elif. 주차구역이면
                        (TODO)
                / elif. 배달구간이면
                        (TODO)
                / elif. 유턴 시작이면
                / elif. 유턴 끝나면
            
            # elif. 미션 구간에 진입하면 (*기준은 Throttle.DISTANCE = 20)
                /if. 정지선이면 (shouldStop  = true, shouldDecel = false)
                /elif. STOP_PARKING 이면 (shouldStop  = true, shouldDecel = false)
                /elif. DECEL 이면 (shouldStop = false, shouldDecSel = true)


        3. traffic_light_status 를 받음
        4?. local path done 을 받음 
        //현재 waypoint 탐색
        calcGlobalCurrWaypoint
        calcLocalCurrWaypoint
        calcParkingCurrWaypoint
        
        //??  유턴지점을 알기 위한 isLeftTurn()
        1. 30개 인덱스를 미리 보면서 return std::abs()
        //

3. pure_pursuit.cpp
헤더에는 controller.h include 
        class PurePursuit: public Controller ????이거 무슨 문법임??
        PurePursuit(char control_target)

        calcGlobalNextWaypoint
        calcLocalNextWaypoint
        calcParkingNextWaypoint

4. controller_node.cpp
헤더에는 pure_pursuit.h include

    void paramCallback()을 통해서
    config의 velocity값을 control->velocity에다가 저장 
    config의 lookahead_distance값을 control->lookahead_distance에다가 저장 

    main문에서는 
    타켓(e 또는 m)에 맞는 control 객체를 생성
    dynamic_reconfigure 이용해서 config값 적용

    while(ros::ok){
        case 1 (if) : 진입 parkinpath 또는 진출 parkingpath가 set인 경우
            bool message_received 가 정의되이있고
            case 1-1 : 전진 path
                if parking_in_idx가 끝까지 진행됐을 경우에
                    3.5초간 정지하는 로직을 수행
                else
                    <<<<<< 주차 관련 velocity, brake, lookahead_distance 설정 >>>>>>>


            case 1-2 : 후진 path 
            
        case 2 (elif) : local path 가 set이면
            control.calcSteering
            control.publishControlMsg

        case 3 (elif) : is_finish가 아니고 path가 set이고 pose가 set이면 global path주행
            control.velocity = control.optimal_throttle.calcTanhVelocity()를 이용해서 
            속도를 미션에 맞게 대응할 수 있도록 조정함 
            

    }
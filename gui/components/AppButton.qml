import QtQuick
import QtQuick.Controls

Button {
    property color tint: Theme.lavender
    property bool primary: false

    implicitHeight: 40
    leftPadding: 18
    rightPadding: 18
    font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.07 : 14
    scale: down ? 0.985 : 1.0

    Behavior on scale { NumberAnimation { duration: 120; easing.type: Easing.OutCubic } }

    contentItem: Text {
        text: parent.text
        color: !parent.enabled
               ? Qt.rgba(Theme.textMuted.r, Theme.textMuted.g, Theme.textMuted.b, 0.65)
               : parent.primary
                 ? Theme.bg0
                 : Theme.textMain
        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.07 : 14
        font.weight: Font.Medium
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    background: Rectangle {
        radius: height / 2
        color: !parent.enabled
               ? Qt.rgba(1, 1, 1, 0.06)
               : parent.primary
                 ? parent.down
                   ? Qt.rgba(parent.tint.r, parent.tint.g, parent.tint.b, 0.96)
                   : parent.hovered
                     ? Qt.rgba(parent.tint.r, parent.tint.g, parent.tint.b, 0.9)
                     : Qt.rgba(parent.tint.r, parent.tint.g, parent.tint.b, 0.82)
                 : parent.down
                  ? Qt.rgba(parent.tint.r, parent.tint.g, parent.tint.b, 0.28)
                  : parent.hovered
                    ? Qt.rgba(parent.tint.r, parent.tint.g, parent.tint.b, 0.22)
                    : Qt.rgba(parent.tint.r, parent.tint.g, parent.tint.b, 0.16)
        border.width: 1
        border.color: !parent.enabled
                      ? Qt.rgba(1, 1, 1, 0.08)
                      : parent.primary
                        ? Qt.rgba(parent.tint.r, parent.tint.g, parent.tint.b, 0.96)
                        : Qt.rgba(parent.tint.r, parent.tint.g, parent.tint.b, 0.34)
        Behavior on color { ColorAnimation { duration: 180; easing.type: Easing.OutCubic } }
        Behavior on border.color { ColorAnimation { duration: 180; easing.type: Easing.OutCubic } }
    }
}

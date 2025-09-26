# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256
import re
import enum

from . import instrument_pb2

enums = {}  # cache enums here for speeding up


def undescore_to_pascal(s):
    sp = s.split("_")
    return "".join([e[0].upper() + e[1:] for e in sp])


def undescore_to_camel(s):
    sp = s.split("_")
    return "".join([sp[0]] + [e[0].upper() + e[1:] for e in sp[1:]])


def to_underscore(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def deserialize(message):
    ret = {}
    for field in message.DESCRIPTOR.fields:  # watchout message.ListFields() does not include defaults
        value = getattr(message, field.name)
        field_name = to_underscore(field.name)
        if field.type == field.TYPE_MESSAGE:
            if field.label == field.LABEL_REPEATED:
                multi = []
                for v in value:
                    multi.append(deserialize(v))
                ret[field_name] = multi
            else:
                ret[field_name] = deserialize(value)
        elif field.type == field.TYPE_ENUM:
            enum_name = field.enum_type.name
            if enum_name not in enums:
                enum_class = getattr(instrument_pb2, enum_name)
                enums[enum_name] = enum.Enum(enum_name, {e[0]: e[1] for e in enum_class.items()})

            if field.label == field.LABEL_REPEATED:
                multi = []
                for v in value:
                    multi.append(enums[enum_name](v))
                ret[field_name] = multi
            else:
                ret[field_name] = enums[enum_name](value)
        else:
            ret[field_name] = value
    return ret


def append_DictFromString(cls):
    def deserialization(*args, **kwargs):
        message = cls.FromString(*args, **kwargs)
        deserialized = deserialize(message)
        if len(deserialized) == 1 and 'value' in deserialized:
            return deserialized['value']
        return deserialized

    cls.DictFromString = deserialization
    return cls


def serialize(kwargs, descriptor):
    message_class = getattr(instrument_pb2, descriptor.name)
    kwargs_transformed = {}
    if type(kwargs) != dict:
        kwargs = {"value": kwargs}
    for field in message_class.DESCRIPTOR.fields:
        field_name = to_underscore(field.name)
        if field.type == field.TYPE_MESSAGE:
            if field.label == field.LABEL_REPEATED:
                multi = []
                for v in kwargs[field_name]:
                    multi.append(serialize(v, getattr(instrument_pb2, field.message_type.name).DESCRIPTOR))
                kwargs_transformed[field.name] = multi
            else:
                kwargs_transformed[field.name] = serialize(kwargs[field_name], getattr(instrument_pb2, field.message_type.name).DESCRIPTOR)
        elif field.type == field.TYPE_ENUM:
            if field.label == field.LABEL_REPEATED:
                kwargs_transformed[field.name] = [e.name for e in kwargs[field_name]]  # assuming generated enums same
            else:
                kwargs_transformed[field.name] = kwargs[field_name].value  # assuming generated enums same
        else:
            kwargs_transformed[field.name] = kwargs[field_name]
    message = message_class(**kwargs_transformed)
    return message

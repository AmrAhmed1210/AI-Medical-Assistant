using AutoMapper;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;

namespace MedicalAssistant.Services.MappingProfiles;

public class DoctorProfile : Profile
{
    public DoctorProfile()
    {
        // Mapping من Doctor Entity إلى DoctorDTO (لعرض القوائم)
        CreateMap<Doctor, DoctorDTO>()
            .ForMember(dest => dest.Specialty, opt => opt.MapFrom(src => src.Specialty.Name))
            .ForMember(dest => dest.ImageUrl, opt => opt.MapFrom(src => src.ImageUrl ?? string.Empty));

        // Mapping من Doctor Entity إلى DoctorDetailsDTO (لعرض التفاصيل)
        CreateMap<Doctor, DoctorDetailsDTO>()
            .IncludeBase<Doctor, DoctorDTO>()  // يستخدم mapping من DoctorDTO
            .ForMember(dest => dest.Bio, opt => opt.MapFrom(src => src.Bio))
            .ForMember(dest => dest.Experience, opt => opt.MapFrom(src => src.Experience));
    }
}
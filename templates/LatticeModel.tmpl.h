//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================


#include "core/DataTypes.h"
#include "core/logging/Logging.h"

#include "field/GhostLayerField.h"
#include "field/SwapableCompare.h"
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "stencil/{{stencilName}}.h"

#include "lbm/lattice_model/EquilibriumDistribution.h"
#include "lbm/field/Density.h"
#include "lbm/field/DensityAndMomentumDensity.h"
#include "lbm/field/DensityAndVelocity.h"
#include "lbm/field/PressureTensor.h"
#include "lbm/field/ShearRate.h"

#include <set>

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#ifdef WALBERLA_CXX_COMPILER_IS_GNU
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif


{% set lmIgnores = ['pdfs', 'pdfs_tmp'] %}


// Forward declarations
namespace walberla{
namespace {{namespace}} {
   class {{className}};
}}
namespace walberla {
namespace mpi {
    mpi::SendBuffer & operator<< (mpi::SendBuffer & buf, const ::walberla::{{namespace}}::{{className}} & lm);
    mpi::RecvBuffer & operator>> (mpi::RecvBuffer & buf,       ::walberla::{{namespace}}::{{className}} & lm);
}}




namespace walberla {
namespace {{namespace}} {


/**
{{className}} was generated with lbmpy. Do not edit this file directly. Instead modify {{className}}.py.
For details see documentation of lbmpy.

Usage:
    - Create an instance of this lattice model class: the constructor parameters vary depending on the configure
      lattice model. A model with constant force needs a single force vector, while a model with variable forces needs
      a force field. All constructor parameters are ordered alphabetically.
    - Create a PDFField with the lattice model as template argument to store the particle distribution functions.
      Use the PDFField to get and modify macroscopic values.
    - The internal class {{className}}::Sweep is a functor to execute one LB time step.
      Stream, collide steps can be executed separately, or together in an optimized stream-pull-collide scheme

*/
class {{className}}
{

public:
    typedef stencil::{{stencilName}} Stencil;
    typedef stencil::{{stencilName}} CommunicationStencil;
    static const real_t w[{{Q}}];
    static const real_t wInv[{{Q}}];

    static const bool compressible = {{compressible}};
    static const int equilibriumAccuracyOrder = {{equilibriumAccuracyOrder}};

    class Sweep
    {
    public:
        Sweep( BlockDataID _pdfsID ) : pdfsID(_pdfsID) {};

        //void stream       ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );
        void collide      ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );
        void streamCollide( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );

        void operator() ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) )
        {
            streamCollide( block, numberOfGhostLayersToInclude );
        }

    private:
        BlockDataID pdfsID;
    };

    {{className}}( {{streamCollideKernel|generateConstructorParameters(lmIgnores) }} )
        : {{ streamCollideKernel|generateConstructorInitializerList(lmIgnores) }}, currentLevel(0)
    {};

    void configure( IBlock & block, StructuredBlockStorage &)  { configureBlock( &block ); }

private:
    void configureBlock(IBlock * block)
    {
        {{streamCollideKernel|generateBlockDataToFieldExtraction(lmIgnores, noDeclarations=True)|indent(8)}}

        {% if refinementLevelScaling -%}
        const uint_t targetLevel = block->getBlockStorage().getLevel(*block);

        if( targetLevel != currentLevel )
        {
            const real_t powTwoTarget = real_c( uint_t(1) << targetLevel );
            const real_t powTwoLevel  = real_c( uint_t(1) << currentLevel );
            const real_t levelScaleFactor = powTwoLevel / powTwoTarget;

            {% for scalingType, name, expression in refinementLevelScaling.scalings -%}
            {% if scalingType == 'normal' %}
            {{name}} = {{expression}};
            {% elif scalingType in ('fieldWithF', 'fieldXYZ') %}
            auto it = {{name}}->{% if scalingType == 'fieldWithF'%} beginWithGhostLayer(){% else %}beginWithGhostLayerXYZ(){% endif %};
            for( ; it != {{name}}->end(); ++it )
            {
                 auto x = it.x();
                 auto y = it.y();
                 auto z = it.z();
                 {% if scalingType == 'fieldWithF' -%}
                 auto f = it.f();
                 {% endif -%}
                 *it = {{expression}};
            }
            {% endif -%}
            {% endfor -%}

            currentLevel = targetLevel;
        }
        {% endif -%}
    }

    // Parameters:
    {{streamCollideKernel|generateMembers(lmIgnores)|indent(4)}}

    // Updated by configureBlock:
    {{streamCollideKernel|generateBlockDataToFieldExtraction(lmIgnores, declarationsOnly=True)|indent(4)}}
    uint_t currentLevel;

    // Backend classes can access private members:
    friend class {{className}}::Sweep;
    template<class LM, class Enable> friend class  EquilibriumDistribution;
    template<class LM, class Enable> friend struct Equilibrium;
    template<class LM, class Enable> friend struct internal::AdaptVelocityToForce;
    template<class LM, class Enable> friend struct Density;
    template<class LM>               friend struct DensityAndVelocity;
    template<class LM, class Enable> friend struct DensityAndMomentumDensity;
    template<class LM, class Enable> friend struct MomentumDensity;
    template<class LM, class It, class Enable> friend struct DensityAndVelocityRange;

    friend mpi::SendBuffer & ::walberla::mpi::operator<< (mpi::SendBuffer & , const {{className}} & );
    friend mpi::RecvBuffer & ::walberla::mpi::operator>> (mpi::RecvBuffer & ,       {{className}} & );

};




//======================================================================================================================
//
//  Implementation of macroscopic value backend
//
//======================================================================================================================



template<>
class EquilibriumDistribution< {{className}}, void>
{
public:
   typedef typename {{className}}::Stencil Stencil;

   static real_t get( const stencil::Direction direction,
                      const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ),
                      real_t rho = real_t(1.0) )
   {
        {% if compressible == 'false' %}
        rho -= real_t(1.0);
        {% endif %}
        {{equilibriumFromDirection}}
   }

   static real_t getSymmetricPart( const stencil::Direction direction,
                                   const Vector3<real_t> & u = Vector3< real_t >(real_t(0.0)),
                                   real_t rho = real_t(1.0) )
   {
        {% if compressible == 'false' %}
        rho -= real_t(1.0);
        {% endif %}
        {{symmetricEquilibriumFromDirection}}
   }

   static real_t getAsymmetricPart( const stencil::Direction direction,
                                    const Vector3< real_t > & u = Vector3<real_t>( real_t(0.0) ),
                                    real_t rho = real_t(1.0) )
   {
        {% if compressible == 'false' %}
        rho -= real_t(1.0);
        {% endif %}
        {{asymmetricEquilibriumFromDirection}}
   }

   static std::vector< real_t > get( const Vector3< real_t > & u = Vector3<real_t>( real_t(0.0) ),
                                     real_t rho = real_t(1.0) )
   {
      {% if compressible == 'false' %}
      rho -= real_t(1.0);
      {% endif %}

      std::vector< real_t > equilibrium( Stencil::Size );
      for( auto d = Stencil::begin(); d != Stencil::end(); ++d )
      {
         equilibrium[d.toIdx()] = get(*d, u, rho);
      }
      return equilibrium;
   }
};


namespace internal {

template<>
struct AdaptVelocityToForce<{{className}}, void>
{
   template< typename FieldPtrOrIterator >
   static Vector3<real_t> get( FieldPtrOrIterator & it, const {{className}} & lm,
                               const Vector3< real_t > & velocity, const real_t rho )
   {
      auto x = it.x();
      auto y = it.y();
      auto z = it.z();
      {% if macroscopicVelocityShift %}
      return velocity - Vector3<real_t>({{macroscopicVelocityShift | join(",") }} {% if D == 2 %}, 0.0 {%endif %} );
      {% else %}
      return velocity;
      {% endif %}
   }

   static Vector3<real_t> get( const cell_idx_t x, const cell_idx_t y, const cell_idx_t z, const {{className}} & lm,
                               const Vector3< real_t > & velocity, const real_t rho )
   {
      {% if macroscopicVelocityShift %}

      return velocity - Vector3<real_t>({{macroscopicVelocityShift | join(",") }} {% if D == 2 %}, 0.0 {%endif %} );
      {% else %}
      return velocity;
      {% endif %}
   }
};
} // namespace internal



template<>
struct Equilibrium< {{className}}, void >
{

   template< typename FieldPtrOrIterator >
   static void set( FieldPtrOrIterator & it,
                    const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rho = real_t(1.0) )
   {
       {% for eqTerm in equilibrium -%}
          it[{{loop.index0 }}] = {{eqTerm}};
       {% endfor -%}

   }

   template< typename PdfField_T >
   static void set( PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z,
                    const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rho = real_t(1.0) )
   {
      real_t & xyz0 = pdf(x,y,z,0);
      {% for eqTerm in equilibrium -%}
         pdf.getF( &xyz0, {{loop.index0 }})= {{eqTerm}};
      {% endfor -%}

   }
};


template<>
struct Density<{{className}}, void>
{
   template< typename FieldPtrOrIterator >
   static inline real_t get( const {{className}} & , const FieldPtrOrIterator & it )
   {
        {% for i in range(Q) -%}
            const real_t f_{{i}} = it[{{i}}];
        {% endfor -%}
        {{densityGetters | indent(8)}}
        return rho;
   }

   template< typename PdfField_T >
   static inline real_t get( const {{className}} & ,
                             const PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
        const real_t & xyz0 = pdf(x,y,z,0);
        {% for i in range(Q) -%}
            const real_t f_{{i}} = pdf.getF( &xyz0, {{i}});
        {% endfor -%}
        {{densityGetters | indent(8)}}
        return rho;
   }
};


template<>
struct DensityAndVelocity<{{className}}>
{
    template< typename FieldPtrOrIterator >
    static void set( FieldPtrOrIterator & it, const {{className}} & lm,
                     const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rhoIn = real_t(1.0) )
    {
        auto x = it.x();
        auto y = it.y();
        auto z = it.z();

        {{densityVelocitySetterMacroscopicValues | indent(8)}}
        {% if D == 2 -%}
        const real_t u_2(0.0);
        {% endif %}

        Equilibrium<{{className}}>::set(it, Vector3<real_t>(u_0, u_1, u_2), rho);
    }

    template< typename PdfField_T >
    static void set( PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z, const {{className}} & lm,
                     const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rhoIn = real_t(1.0) )
    {
        {{densityVelocitySetterMacroscopicValues | indent(8)}}
        {% if D == 2 -%}
        const real_t u_2(0.0);
        {% endif %}


        Equilibrium<{{className}}>::set(pdf, x, y, z, Vector3<real_t>(u_0, u_1, u_2), rho);
    }
};


template<typename FieldIteratorXYZ >
struct DensityAndVelocityRange<{{className}}, FieldIteratorXYZ>
{

   static void set( FieldIteratorXYZ & begin, const FieldIteratorXYZ & end, const {{className}} & lm,
                    const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rhoIn = real_t(1.0) )
   {
        for( auto cellIt = begin; cellIt != end; ++cellIt )
        {
            const auto x = cellIt.x();
            const auto y = cellIt.y();
            const auto z = cellIt.z();
            {{densityVelocitySetterMacroscopicValues | indent(12)}}
            {% if D == 2 -%}
            const real_t u_2(0.0);
            {% endif %}

            Equilibrium<{{className}}>::set(cellIt, Vector3<real_t>(u_0, u_1, u_2), rho);
        }
   }
};



template<>
struct DensityAndMomentumDensity<{{className}}>
{
   template< typename FieldPtrOrIterator >
   static real_t get( Vector3< real_t > & momentumDensity, const {{className}} & lm,
                      const FieldPtrOrIterator & it )
   {
        const auto x = it.x();
        const auto y = it.y();
        const auto z = it.z();

        {% for i in range(Q) -%}
            const real_t f_{{i}} = it[{{i}}];
        {% endfor -%}

        {{momentumDensityGetter | indent(8) }}
        {% for i in range(D) -%}
            momentumDensity[{{i}}] = md_{{i}};
        {% endfor %}
        return rho;
   }

   template< typename PdfField_T >
   static real_t get( Vector3< real_t > & momentumDensity, const {{className}} & lm, const PdfField_T & pdf,
                      const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
        const real_t & xyz0 = pdf(x,y,z,0);
        {% for i in range(Q) -%}
            const real_t f_{{i}} = pdf.getF( &xyz0, {{i}});
        {% endfor -%}

        {{momentumDensityGetter | indent(8) }}
        {% for i in range(D) -%}
            momentumDensity[{{i}}] = md_{{i}};
        {% endfor %}
       return rho;
   }
};


template<>
struct MomentumDensity< {{className}}>
{
   template< typename FieldPtrOrIterator >
   static void get( Vector3< real_t > & momentumDensity, const {{className}} & lm, const FieldPtrOrIterator & it )
   {
        const auto x = it.x();
        const auto y = it.y();
        const auto z = it.z();

        {% for i in range(Q) -%}
            const real_t f_{{i}} = it[{{i}}];
        {% endfor -%}

        {{momentumDensityGetter | indent(8) }}
        {% for i in range(D) -%}
            momentumDensity[{{i}}] = md_{{i}};
        {% endfor %}
   }

   template< typename PdfField_T >
   static void get( Vector3< real_t > & momentumDensity, const {{className}} & lm, const PdfField_T & pdf,
                    const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
        const real_t & xyz0 = pdf(x,y,z,0);
        {% for i in range(Q) -%}
            const real_t f_{{i}} = pdf.getF( &xyz0, {{i}});
        {% endfor -%}

        {{momentumDensityGetter | indent(8) }}
        {% for i in range(D) -%}
            momentumDensity[{{i}}] = md_{{i}};
        {% endfor %}
   }
};


template<>
struct PressureTensor<{{className}}>
{
   template< typename FieldPtrOrIterator >
   static void get( Matrix3< real_t > & /* pressureTensor */, const {{className}} & /* latticeModel */, const FieldPtrOrIterator & /* it */ )
   {
       WALBERLA_ABORT("Not implemented");
   }

   template< typename PdfField_T >
   static void get( Matrix3< real_t > & /* pressureTensor */, const {{className}} & /* latticeModel */, const PdfField_T & /* pdf */,
                    const cell_idx_t /* x */, const cell_idx_t /* y */, const cell_idx_t /* z */ )
   {
       WALBERLA_ABORT("Not implemented");
   }
};


template<>
struct ShearRate<{{className}}>
{
   template< typename FieldPtrOrIterator >
   static inline real_t get( const {{className}} & /* latticeModel */, const FieldPtrOrIterator & /* it */,
                             const Vector3< real_t > & /* velocity */, const real_t /* rho */)
   {
       WALBERLA_ABORT("Not implemented");
       return real_t(0.0);
   }

   template< typename PdfField_T >
   static inline real_t get( const {{className}} & latticeModel,
                             const PdfField_T & /* pdf */, const cell_idx_t /* x */, const cell_idx_t /* y */, const cell_idx_t /* z */,
                             const Vector3< real_t > & /* velocity */, const real_t /* rho */ )
   {
       WALBERLA_ABORT("Not implemented");
       return real_t(0.0);
   }

   static inline real_t get( const std::vector< real_t > & /* nonEquilibrium */, const real_t /* relaxationParam */,
                             const real_t /* rho */ = real_t(1) )
   {
       WALBERLA_ABORT("Not implemented");
       return real_t(0.0);
   }
};


} // namespace {{namespace}}
} // namespace walberla



#ifdef WALBERLA_CXX_COMPILER_IS_GNU
#pragma GCC diagnostic pop
#endif